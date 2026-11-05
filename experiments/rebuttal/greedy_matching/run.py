#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from joblib import effective_n_jobs
from omegaconf import OmegaConf

# Each process works on an independent target/pair.  Prevent BLAS from starting
# another large thread pool inside every worker.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

from experiments.universality.shared import filter_models, load_components_for_rank

from experiments.rebuttal.greedy_matching.plot import make_figure

logger = logging.getLogger("greedy_matching")

N_PERMS = 1000
BOOTSTRAP = 10000


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(
        path, json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n"
    )


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".csv", delete=False
    ) as handle:
        temporary = Path(handle.name)
    frame.to_csv(temporary, index=False, float_format="%.10g")
    os.replace(temporary, path)


def _atomic_savez(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _factor_inventory(
    data_root: Path, dataset: str, rank: int, models: list[str]
) -> tuple[list[str], list[dict[str, Any]], Path]:
    """Resolve the exact canonical sigma-selected factor files.

    This mirrors ``load_components_for_rank`` before loading the large arrays,
    which also lets completed runs reuse their normalized-factor cache.
    """
    embeddings_dir = data_root / "embeddings" / dataset
    selection_path = data_root / "meta" / dataset / f"bandwidth_selection_rank_{rank}.csv"
    if not selection_path.exists():
        raise FileNotFoundError(f"Missing canonical sigma selection: {selection_path}")

    selection = pd.read_csv(selection_path)
    required = {"model", "sigma_mult", "best_seed"}
    missing = required.difference(selection.columns)
    if missing:
        raise ValueError(f"{selection_path} is missing columns: {sorted(missing)}")
    selection_by_model = selection.set_index("model")

    # Match the canonical paper computation's model order (metadata order),
    # rather than filesystem/alphabetical order.  This matters for deterministic
    # source subsampling in the permutation null.
    metadata_path = data_root / "meta" / dataset / "model_metadata.csv"
    metadata = pd.read_csv(metadata_path)
    if "feature_uid" not in metadata:
        raise ValueError(f"{metadata_path} is missing feature_uid")
    candidates = filter_models(metadata["feature_uid"].astype(str).tolist())
    if models:
        requested = set(models)
        candidates = [name for name in candidates if name in requested]
    candidates = [
        name
        for name in candidates
        if (embeddings_dir / name / f"rank_{rank}").is_dir()
    ]
    model_names = [name for name in candidates if name in selection_by_model.index]
    if len(model_names) < 2:
        raise ValueError(f"Need at least two models, found {len(model_names)}")

    records: list[dict[str, Any]] = []
    for name in model_names:
        row = selection_by_model.loc[name]
        factor_dir = (
            embeddings_dir
            / name
            / f"rank_{rank}"
            / f"sigma_{row['sigma_mult']}"
            / f"seed_{int(row['best_seed'])}"
        )
        factor_path = factor_dir / "w.npz"
        if not factor_path.exists():
            factor_path = factor_dir / "w.npy"
        if not factor_path.exists():
            raise FileNotFoundError(f"Missing selected factor for {name}: {factor_dir}")
        stat = factor_path.stat()
        records.append(
            {
                "model": name,
                "path": str(factor_path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sigma_mult": float(row["sigma_mult"]),
                "best_seed": int(row["best_seed"]),
            }
        )
    return model_names, records, selection_path


def _run_signature(
    rank: int,
    model_names: list[str],
    records: list[dict[str, Any]],
    selection_path: Path,
) -> str:
    payload = {
        # Factor-cache schema 2: the normalized inputs did not change when the
        # third matching rule was added, so the signature stays stable.
        "schema": 2,
        "rank": rank,
        "models": model_names,
        "factors": records,
        "selection_sha256": _sha256(selection_path),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    return f"rank{rank}_n{len(model_names)}_{digest}"


def _normalized_cache_is_valid(
    path: Path, manifest_path: Path, signature: str, n_models: int, rank: int
) -> bool:
    if not path.exists() or not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("signature") != signature:
            return False
        array = np.load(path, mmap_mode="r")
        return (
            array.ndim == 3
            and array.shape[0] == n_models
            and array.shape[2] == rank
            and array.dtype == np.float64
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _prepare_normalized_factors(
    *,
    rank: int,
    dataset: str,
    model_names: list[str],
    cache_dir: Path,
    signature: str,
) -> tuple[Path, tuple[int, int, int]]:
    normalized_path = cache_dir / "normalized_w.npy"
    manifest_path = cache_dir / "normalized_manifest.json"
    if _normalized_cache_is_valid(
        normalized_path, manifest_path, signature, len(model_names), rank
    ):
        array = np.load(normalized_path, mmap_mode="r")
        logger.info("Reusing normalized factors %s with shape %s", normalized_path, array.shape)
        return normalized_path, tuple(array.shape)

    logger.info("Loading the canonical sigma-selected factors")
    components = load_components_for_rank(
        rank=rank,
        models=model_names,
        dataset=dataset,
        exclude_trajectory=True,
    )
    loaded_names = list(components)
    if loaded_names != model_names:
        missing = sorted(set(model_names).difference(loaded_names))
        extra = sorted(set(loaded_names).difference(model_names))
        raise RuntimeError(f"Canonical factor loader mismatch; missing={missing}, extra={extra}")

    first = np.asarray(components[model_names[0]])
    if first.ndim != 2 or first.shape[1] != rank:
        raise ValueError(f"Unexpected factor shape for {model_names[0]}: {first.shape}")
    n_items = first.shape[0]
    shape = (len(model_names), n_items, rank)
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = cache_dir / f".normalized_w_{os.getpid()}.npy"
    normalized = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float64, shape=shape
    )
    try:
        for index, name in enumerate(model_names):
            matrix = np.asarray(components.pop(name), dtype=np.float64)
            if matrix.shape != (n_items, rank):
                raise ValueError(f"Inconsistent factor shape for {name}: {matrix.shape}")
            if not np.isfinite(matrix).all():
                raise ValueError(f"Non-finite values in factor matrix for {name}")
            norms = np.linalg.norm(matrix, axis=0, keepdims=True)
            normalized[index] = matrix / (norms + 1e-8)
            if (index + 1) % 20 == 0 or index + 1 == len(model_names):
                logger.info("  normalized %d/%d models", index + 1, len(model_names))
        normalized.flush()
        del normalized
        os.replace(temporary, normalized_path)
    finally:
        temporary.unlink(missing_ok=True)

    _atomic_write_json(
        manifest_path,
        {
            "schema_version": 2,
            "signature": signature,
            "shape": shape,
            "dtype": "float64",
            "normalization": "columnwise L2 with denominator norm + 1e-8",
        },
    )
    return normalized_path, shape


def _sequential_greedy(
    cosine_squared: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedily select the best remaining edge in a square score matrix."""
    n_rows, n_columns = cosine_squared.shape
    if n_rows != n_columns:
        raise ValueError(
            "Sequential greedy one-to-one matching requires a square matrix; "
            f"received {cosine_squared.shape}"
        )
    # Stable sorting gives deterministic row-major tie breaking.
    edge_order = np.argsort(-cosine_squared.ravel(), kind="stable")
    row_used = np.zeros(n_rows, dtype=bool)
    column_used = np.zeros(n_columns, dtype=bool)
    assignment = np.full(n_rows, -1, dtype=np.int16)
    n_selected = 0
    for flat_index in edge_order:
        row = int(flat_index // n_columns)
        column = int(flat_index % n_columns)
        if row_used[row] or column_used[column]:
            continue
        assignment[row] = column
        row_used[row] = True
        column_used[column] = True
        n_selected += 1
        if n_selected == n_rows:
            break
    if np.any(assignment < 0):
        raise RuntimeError("Sequential greedy matching failed to assign every row")
    scores = cosine_squared[np.arange(n_rows), assignment]
    return scores.astype(np.float64), assignment


def _score_pair(
    target_normalized: np.ndarray, source_normalized: np.ndarray
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return scores and assignments for all three matching rules."""
    cosine_squared = (target_normalized.T @ source_normalized) ** 2
    independent_assignment = np.argmax(cosine_squared, axis=1).astype(np.int16)
    independent = cosine_squared[
        np.arange(cosine_squared.shape[0]), independent_assignment
    ].astype(np.float64)
    sequential, sequential_assignment = _sequential_greedy(cosine_squared)
    row_indices, column_indices = linear_sum_assignment(-cosine_squared)
    hungarian = np.empty(cosine_squared.shape[0], dtype=np.float64)
    hungarian[row_indices] = cosine_squared[row_indices, column_indices]
    return (
        independent,
        sequential,
        hungarian,
        independent_assignment,
        sequential_assignment,
    )


def _valid_raw_base_checkpoint(
    path: Path, target_index: int, n_models: int, rank: int
) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return (
                int(data["target_index"]) == target_index
                and data["greedy"].shape == (n_models, rank)
                and data["hungarian"].shape == (n_models, rank)
                and data["greedy_assignment"].shape == (n_models, rank)
            )
    except (OSError, ValueError, KeyError):
        return False


def _valid_raw_checkpoint(path: Path, target_index: int, n_models: int, rank: int) -> bool:
    if not _valid_raw_base_checkpoint(path, target_index, n_models, rank):
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return (
                data["sequential_greedy"].shape == (n_models, rank)
                and data["sequential_greedy_assignment"].shape == (n_models, rank)
            )
    except (OSError, ValueError, KeyError):
        return False


def _raw_target_worker(
    target_index: int,
    normalized_path: str,
    checkpoint_path: str,
    n_models: int,
    rank: int,
    reuse_base_checkpoint: bool,
) -> int:
    all_normalized = np.load(normalized_path, mmap_mode="r")
    target = np.ascontiguousarray(all_normalized[target_index], dtype=np.float64)
    if reuse_base_checkpoint:
        with np.load(checkpoint_path, allow_pickle=False) as data:
            independent = data["greedy"].astype(np.float64)
            hungarian = data["hungarian"].astype(np.float64)
            independent_assignment = data["greedy_assignment"].astype(np.int16)
    else:
        independent = np.full((n_models, rank), np.nan, dtype=np.float64)
        hungarian = np.full((n_models, rank), np.nan, dtype=np.float64)
        independent_assignment = np.full((n_models, rank), -1, dtype=np.int16)
    sequential = np.full((n_models, rank), np.nan, dtype=np.float64)
    sequential_assignment = np.full((n_models, rank), -1, dtype=np.int16)
    for source_index in range(n_models):
        if source_index == target_index:
            continue
        source = np.ascontiguousarray(all_normalized[source_index], dtype=np.float64)
        if reuse_base_checkpoint:
            cosine_squared = (target.T @ source) ** 2
            s_score, s_assignment = _sequential_greedy(cosine_squared)
        else:
            (
                i_score,
                s_score,
                h_score,
                i_assignment,
                s_assignment,
            ) = _score_pair(target, source)
            independent[source_index] = i_score
            hungarian[source_index] = h_score
            independent_assignment[source_index] = i_assignment
        sequential[source_index] = s_score
        sequential_assignment[source_index] = s_assignment
    _atomic_savez(
        Path(checkpoint_path),
        target_index=np.asarray(target_index, dtype=np.int32),
        greedy=independent,
        sequential_greedy=sequential,
        hungarian=hungarian,
        greedy_assignment=independent_assignment,
        sequential_greedy_assignment=sequential_assignment,
    )
    return target_index


def _compute_raw_scores(
    *,
    normalized_path: Path,
    raw_dir: Path,
    n_models: int,
    rank: int,
    n_jobs: int,
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    pending: list[tuple[int, Path, bool]] = []
    for target_index in range(n_models):
        checkpoint = raw_dir / f"target_{target_index:03d}.npz"
        if not _valid_raw_checkpoint(checkpoint, target_index, n_models, rank):
            reuse_base_checkpoint = _valid_raw_base_checkpoint(
                checkpoint, target_index, n_models, rank
            )
            pending.append((target_index, checkpoint, reuse_base_checkpoint))
    logger.info(
        "Raw scores: %d/%d target checkpoints already complete",
        n_models - len(pending),
        n_models,
    )
    if not pending:
        return

    workers = max(1, min(effective_n_jobs(n_jobs), len(pending)))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _raw_target_worker,
                target_index,
                str(normalized_path),
                str(checkpoint),
                n_models,
                rank,
                reuse_base_checkpoint,
            ): target_index
            for target_index, checkpoint, reuse_base_checkpoint in pending
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            target_index = future.result()
            if completed % max(1, len(pending) // 10) == 0 or completed == len(pending):
                logger.info(
                    "  raw progress %d/%d (target %d)",
                    completed,
                    len(pending),
                    target_index,
                )


def _assemble_raw(
    raw_dir: Path, n_models: int, rank: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    independent = np.full((n_models, n_models, rank), np.nan, dtype=np.float64)
    sequential = np.full_like(independent, np.nan)
    hungarian = np.full_like(independent, np.nan)
    independent_assignment = np.full(
        (n_models, n_models, rank), -1, dtype=np.int16
    )
    sequential_assignment = np.full_like(independent_assignment, -1)
    for target_index in range(n_models):
        checkpoint = raw_dir / f"target_{target_index:03d}.npz"
        if not _valid_raw_checkpoint(checkpoint, target_index, n_models, rank):
            raise RuntimeError(f"Invalid or missing raw checkpoint: {checkpoint}")
        with np.load(checkpoint, allow_pickle=False) as data:
            independent[target_index] = data["greedy"]
            sequential[target_index] = data["sequential_greedy"]
            hungarian[target_index] = data["hungarian"]
            independent_assignment[target_index] = data["greedy_assignment"]
            sequential_assignment[target_index] = data[
                "sequential_greedy_assignment"
            ]
    return (
        independent,
        sequential,
        hungarian,
        independent_assignment,
        sequential_assignment,
    )


def _all_sources(n_models: int) -> list[list[int]]:
    """Return every source for every target, excluding the target itself."""
    return [
        [source for source in range(n_models) if source != target]
        for target in range(n_models)
    ]


def _valid_null_base_checkpoint(
    path: Path,
    rank: int,
    n_perms: int,
    target_index: int,
    source_index: int,
    pair_seed: int,
) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return (
                int(data["n_perms"]) == n_perms
                and int(data["target_index"]) == target_index
                and int(data["source_index"]) == source_index
                and int(data["pair_seed"]) == pair_seed
                and data["greedy_p95"].shape == (rank,)
                and data["hungarian_p95"].shape == (rank,)
            )
    except (OSError, ValueError, KeyError):
        return False


def _valid_null_checkpoint(
    path: Path,
    rank: int,
    n_perms: int,
    target_index: int,
    source_index: int,
    pair_seed: int,
) -> bool:
    if not _valid_null_base_checkpoint(
        path,
        rank,
        n_perms,
        target_index,
        source_index,
        pair_seed,
    ):
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return data["sequential_greedy_p95"].shape == (rank,)
    except (OSError, ValueError, KeyError):
        return False


def _null_pair_worker(
    target_index: int,
    source_index: int,
    normalized_path: str,
    checkpoint_path: str,
    rank: int,
    n_perms: int,
    pair_seed: int,
    reuse_base_checkpoint: bool,
) -> tuple[int, int]:
    all_normalized = np.load(normalized_path, mmap_mode="r")
    target = np.ascontiguousarray(all_normalized[target_index], dtype=np.float32)
    source = np.ascontiguousarray(all_normalized[source_index], dtype=np.float32)
    target_transpose = target.T
    n_items = target.shape[0]
    independent_null = (
        None
        if reuse_base_checkpoint
        else np.empty((n_perms, rank), dtype=np.float32)
    )
    sequential_null = np.empty((n_perms, rank), dtype=np.float32)
    hungarian_null = (
        None
        if reuse_base_checkpoint
        else np.empty((n_perms, rank), dtype=np.float32)
    )
    rng = np.random.default_rng(pair_seed)

    for permutation_index in range(n_perms):
        row_permutation = rng.permutation(n_items)
        # Row permutations preserve column norms, so the normalized matrices can
        # be multiplied directly. All methods consume this same matrix.
        cosine_squared = (target_transpose @ source[row_permutation]) ** 2
        sequential_scores, _ = _sequential_greedy(cosine_squared)
        sequential_null[permutation_index] = sequential_scores
        if not reuse_base_checkpoint:
            assert independent_null is not None
            assert hungarian_null is not None
            independent_null[permutation_index] = cosine_squared.max(axis=1)
            rows, columns = linear_sum_assignment(-cosine_squared)
            scores = np.empty(rank, dtype=np.float32)
            scores[rows] = cosine_squared[rows, columns]
            hungarian_null[permutation_index] = scores

    if reuse_base_checkpoint:
        with np.load(checkpoint_path, allow_pickle=False) as data:
            independent_p95 = data["greedy_p95"].astype(np.float32)
            hungarian_p95 = data["hungarian_p95"].astype(np.float32)
    else:
        assert independent_null is not None
        assert hungarian_null is not None
        independent_p95 = np.percentile(
            independent_null, 95, axis=0
        ).astype(np.float32)
        hungarian_p95 = np.percentile(
            hungarian_null, 95, axis=0
        ).astype(np.float32)

    _atomic_savez(
        Path(checkpoint_path),
        target_index=np.asarray(target_index, dtype=np.int32),
        source_index=np.asarray(source_index, dtype=np.int32),
        n_perms=np.asarray(n_perms, dtype=np.int32),
        pair_seed=np.asarray(pair_seed, dtype=np.uint64),
        greedy_p95=independent_p95,
        sequential_greedy_p95=np.percentile(
            sequential_null, 95, axis=0
        ).astype(np.float32),
        hungarian_p95=hungarian_p95,
    )
    return target_index, source_index


def _compute_null_thresholds(
    *,
    normalized_path: Path,
    null_dir: Path,
    n_models: int,
    rank: int,
    n_perms: int,
    seed: int,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[int]]]:
    null_dir.mkdir(parents=True, exist_ok=True)
    sources_by_target = _all_sources(n_models)
    pending: list[tuple[int, int, Path, int, bool]] = []
    total = n_models * (n_models - 1)
    for target_index, sources in enumerate(sources_by_target):
        for source_index in sources:
            checkpoint = (
                null_dir / f"target_{target_index:03d}_source_{source_index:03d}.npz"
            )
            seed_sequence = np.random.SeedSequence(
                [seed, target_index, source_index, n_perms]
            )
            pair_seed = int(seed_sequence.generate_state(1, dtype=np.uint64)[0])
            if not _valid_null_checkpoint(
                checkpoint,
                rank,
                n_perms,
                target_index,
                source_index,
                pair_seed,
            ):
                reuse_base_checkpoint = _valid_null_base_checkpoint(
                    checkpoint,
                    rank,
                    n_perms,
                    target_index,
                    source_index,
                    pair_seed,
                )
                pending.append(
                    (
                        target_index,
                        source_index,
                        checkpoint,
                        pair_seed,
                        reuse_base_checkpoint,
                    )
                )
    logger.info(
        "Null calibration: %d/%d ordered-pair checkpoints already complete",
        total - len(pending),
        total,
    )

    if pending:
        workers = max(1, min(effective_n_jobs(n_jobs), len(pending)))
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _null_pair_worker,
                    target_index,
                    source_index,
                    str(normalized_path),
                    str(checkpoint),
                    rank,
                    n_perms,
                    pair_seed,
                    reuse_base_checkpoint,
                ): (target_index, source_index)
                for (
                    target_index,
                    source_index,
                    checkpoint,
                    pair_seed,
                    reuse_base_checkpoint,
                ) in pending
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                target_index, source_index = future.result()
                if (
                    completed % max(1, len(pending) // 10) == 0
                    or completed == len(pending)
                ):
                    logger.info(
                        "  null progress %d/%d (target %d, source %d)",
                        completed,
                        len(pending),
                        target_index,
                        source_index,
                    )

    shape = (n_models, n_models, rank)
    independent_threshold = np.full(shape, np.nan, dtype=np.float32)
    sequential_threshold = np.full(shape, np.nan, dtype=np.float32)
    hungarian_threshold = np.full(shape, np.nan, dtype=np.float32)
    for target_index, sources in enumerate(sources_by_target):
        for source_index in sources:
            checkpoint = (
                null_dir / f"target_{target_index:03d}_source_{source_index:03d}.npz"
            )
            seed_sequence = np.random.SeedSequence(
                [seed, target_index, source_index, n_perms]
            )
            pair_seed = int(seed_sequence.generate_state(1, dtype=np.uint64)[0])
            if not _valid_null_checkpoint(
                checkpoint,
                rank,
                n_perms,
                target_index,
                source_index,
                pair_seed,
            ):
                raise RuntimeError(f"Invalid or missing null checkpoint: {checkpoint}")
            with np.load(checkpoint, allow_pickle=False) as data:
                independent_threshold[target_index, source_index] = data["greedy_p95"]
                sequential_threshold[target_index, source_index] = data[
                    "sequential_greedy_p95"
                ]
                hungarian_threshold[target_index, source_index] = data["hungarian_p95"]
    return (
        independent_threshold,
        sequential_threshold,
        hungarian_threshold,
        sources_by_target,
    )


def _null_adjust(
    raw: np.ndarray, threshold: np.ndarray
) -> np.ndarray:
    if raw.shape != threshold.shape:
        raise ValueError(
            "raw scores and pair-specific null thresholds must have identical "
            f"shapes, got {raw.shape} and {threshold.shape}"
        )
    adjusted = (raw - threshold) / np.maximum(1.0 - threshold, 1e-8)
    result = np.clip(adjusted, 0.0, 1.0).astype(np.float32)
    indices = np.arange(raw.shape[0])
    result[indices, indices] = np.nan
    return result


def _selection_counts(assignments: np.ndarray, rank: int) -> np.ndarray:
    n_models = assignments.shape[0]
    counts = np.zeros((n_models, n_models, rank), dtype=np.int16)
    for target_index in range(n_models):
        for source_index in range(n_models):
            if target_index == source_index:
                continue
            counts[target_index, source_index] = np.bincount(
                assignments[target_index, source_index], minlength=rank
            )
    return counts


def _rank_descending(series: pd.Series) -> pd.Series:
    return series.rank(method="average", ascending=False)


def _per_dimension_table(
    *,
    model_names: list[str],
    rank: int,
    independent_raw: np.ndarray,
    sequential_greedy_raw: np.ndarray,
    hungarian_raw: np.ndarray,
    independent_adjusted: np.ndarray,
    sequential_greedy_adjusted: np.ndarray,
    hungarian_adjusted: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target_index, model in enumerate(model_names):
        valid_sources = np.arange(len(model_names)) != target_index
        i_raw = independent_raw[target_index, valid_sources].mean(axis=0)
        s_raw = sequential_greedy_raw[target_index, valid_sources].mean(axis=0)
        h_raw = hungarian_raw[target_index, valid_sources].mean(axis=0)
        i_adjusted = independent_adjusted[target_index, valid_sources].mean(axis=0)
        s_adjusted = sequential_greedy_adjusted[
            target_index, valid_sources
        ].mean(axis=0)
        h_adjusted = hungarian_adjusted[target_index, valid_sources].mean(axis=0)
        for dimension in range(rank):
            rows.append(
                {
                    "model": model,
                    "dimension": dimension,
                    "rank": rank,
                    "independent_raw": float(i_raw[dimension]),
                    "sequential_greedy_raw": float(s_raw[dimension]),
                    "hungarian_raw": float(h_raw[dimension]),
                    "delta_independent_minus_hungarian_raw": float(
                        i_raw[dimension] - h_raw[dimension]
                    ),
                    "delta_sequential_greedy_minus_hungarian_raw": float(
                        s_raw[dimension] - h_raw[dimension]
                    ),
                    "independent_adjusted": float(i_adjusted[dimension]),
                    "sequential_greedy_adjusted": float(s_adjusted[dimension]),
                    "hungarian_adjusted": float(h_adjusted[dimension]),
                    "delta_independent_minus_hungarian_adjusted": float(
                        i_adjusted[dimension] - h_adjusted[dimension]
                    ),
                    "delta_sequential_greedy_minus_hungarian_adjusted": float(
                        s_adjusted[dimension] - h_adjusted[dimension]
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    for suffix in ("raw", "adjusted"):
        frame[f"independent_global_rank_{suffix}"] = _rank_descending(
            frame[f"independent_{suffix}"]
        )
        frame[f"sequential_greedy_global_rank_{suffix}"] = _rank_descending(
            frame[f"sequential_greedy_{suffix}"]
        )
        frame[f"hungarian_global_rank_{suffix}"] = _rank_descending(
            frame[f"hungarian_{suffix}"]
        )
        frame[f"rank_shift_independent_minus_hungarian_{suffix}"] = (
            frame[f"independent_global_rank_{suffix}"]
            - frame[f"hungarian_global_rank_{suffix}"]
        )
        frame[f"rank_shift_sequential_greedy_minus_hungarian_{suffix}"] = (
            frame[f"sequential_greedy_global_rank_{suffix}"]
            - frame[f"hungarian_global_rank_{suffix}"]
        )
    return frame


def _pair_table(
    *,
    model_names: list[str],
    rank: int,
    independent_raw: np.ndarray,
    sequential_greedy_raw: np.ndarray,
    hungarian_raw: np.ndarray,
    independent_adjusted: np.ndarray,
    sequential_greedy_adjusted: np.ndarray,
    hungarian_adjusted: np.ndarray,
    selection_counts: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target_index, target in enumerate(model_names):
        for source_index, source in enumerate(model_names):
            if target_index == source_index:
                continue
            counts = selection_counts[target_index, source_index]
            unique = int(np.count_nonzero(counts))
            rows.append(
                {
                    "target_model": target,
                    "source_model": source,
                    "rank": rank,
                    "independent_raw": float(
                        independent_raw[target_index, source_index].mean()
                    ),
                    "sequential_greedy_raw": float(
                        sequential_greedy_raw[target_index, source_index].mean()
                    ),
                    "hungarian_raw": float(
                        hungarian_raw[target_index, source_index].mean()
                    ),
                    "delta_independent_minus_hungarian_raw": float(
                        (
                            independent_raw[target_index, source_index]
                            - hungarian_raw[target_index, source_index]
                        ).mean()
                    ),
                    "delta_sequential_greedy_minus_hungarian_raw": float(
                        (
                            sequential_greedy_raw[target_index, source_index]
                            - hungarian_raw[target_index, source_index]
                        ).mean()
                    ),
                    "independent_adjusted": float(
                        independent_adjusted[target_index, source_index].mean()
                    ),
                    "sequential_greedy_adjusted": float(
                        sequential_greedy_adjusted[
                            target_index, source_index
                        ].mean()
                    ),
                    "hungarian_adjusted": float(
                        hungarian_adjusted[target_index, source_index].mean()
                    ),
                    "delta_independent_minus_hungarian_adjusted": float(
                        (
                            independent_adjusted[target_index, source_index]
                            - hungarian_adjusted[target_index, source_index]
                        ).mean()
                    ),
                    "delta_sequential_greedy_minus_hungarian_adjusted": float(
                        (
                            sequential_greedy_adjusted[
                                target_index, source_index
                            ]
                            - hungarian_adjusted[target_index, source_index]
                        ).mean()
                    ),
                    "n_unique_source_dimensions": unique,
                    "unused_source_fraction": float(1.0 - unique / rank),
                    "target_dimensions_in_collision_fraction": float(
                        counts[counts > 1].sum() / rank
                    ),
                    "maximum_claims_on_one_source_dimension": int(counts.max()),
                    "mean_claims_per_selected_source_dimension": float(rank / unique),
                }
            )
    return pd.DataFrame(rows)


def _broad_dimension_table(
    model_names: list[str], selection_counts: np.ndarray
) -> pd.DataFrame:
    n_models, _, rank = selection_counts.shape
    rows: list[dict[str, Any]] = []
    for source_index, source in enumerate(model_names):
        valid_targets = np.arange(n_models) != source_index
        for source_dimension in range(rank):
            counts = selection_counts[
                valid_targets, source_index, source_dimension
            ].astype(int)
            rows.append(
                {
                    "source_model": source,
                    "source_dimension": source_dimension,
                    "total_claims": int(counts.sum()),
                    "mean_claims_per_target_model": float(counts.mean()),
                    "target_model_coverage": float(np.mean(counts > 0)),
                    "unused_pair_fraction": float(np.mean(counts == 0)),
                    "collision_pair_fraction": float(np.mean(counts > 1)),
                    "excess_claims": int(np.maximum(counts - 1, 0).sum()),
                    "maximum_claims_in_one_pair": int(counts.max()),
                }
            )
    frame = pd.DataFrame(rows)
    frame["coverage_rank_within_source_model"] = frame.groupby("source_model")[
        "target_model_coverage"
    ].rank(method="min", ascending=False)
    frame["coverage_global_rank"] = _rank_descending(frame["target_model_coverage"])
    return frame.sort_values(
        ["target_model_coverage", "total_claims"],
        ascending=[False, False],
        ignore_index=True,
    )


def _selection_distribution(
    selection_counts: np.ndarray,
) -> pd.DataFrame:
    n_models = selection_counts.shape[0]
    off_diagonal = ~np.eye(n_models, dtype=bool)
    values = selection_counts[off_diagonal].ravel()
    claims, frequencies = np.unique(values, return_counts=True)
    return pd.DataFrame(
        {
            "selection_count": claims.astype(int),
            "n_source_dimensions_across_pairs": frequencies.astype(int),
            "fraction": frequencies / frequencies.sum(),
        }
    )


def _gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.allclose(values.sum(), 0):
        return 0.0
    absolute_differences = np.abs(values[:, None] - values[None, :]).sum()
    return float(absolute_differences / (2 * values.size * values.sum()))


def _per_model_table(
    *,
    model_names: list[str],
    per_dimension: pd.DataFrame,
    pairs: pd.DataFrame,
    broad_dimensions: pd.DataFrame,
) -> pd.DataFrame:
    numeric_columns = [
        "independent_raw",
        "sequential_greedy_raw",
        "hungarian_raw",
        "delta_independent_minus_hungarian_raw",
        "delta_sequential_greedy_minus_hungarian_raw",
        "independent_adjusted",
        "sequential_greedy_adjusted",
        "hungarian_adjusted",
        "delta_independent_minus_hungarian_adjusted",
        "delta_sequential_greedy_minus_hungarian_adjusted",
    ]
    means = per_dimension.groupby("model", sort=False)[numeric_columns].mean()
    target_collision = pairs.groupby("target_model", sort=False).agg(
        mean_unused_source_fraction_as_target=("unused_source_fraction", "mean"),
        median_unused_source_fraction_as_target=("unused_source_fraction", "median"),
        mean_target_dimensions_in_collision_fraction=(
            "target_dimensions_in_collision_fraction",
            "mean",
        ),
    )
    source_collision = pairs.groupby("source_model", sort=False).agg(
        mean_unused_source_fraction_as_source=("unused_source_fraction", "mean")
    )
    rows: list[dict[str, Any]] = []
    for model in model_names:
        row: dict[str, Any] = {"model": model}
        row.update(means.loc[model].to_dict())
        row.update(target_collision.loc[model].to_dict())
        row.update(source_collision.loc[model].to_dict())
        source_breadth = broad_dimensions[broad_dimensions["source_model"] == model]
        row["selection_load_gini"] = _gini(
            source_breadth["total_claims"].to_numpy()
        )
        row["maximum_dimension_target_model_coverage"] = float(
            source_breadth["target_model_coverage"].max()
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    for suffix in ("raw", "adjusted"):
        frame[f"independent_model_rank_{suffix}"] = _rank_descending(
            frame[f"independent_{suffix}"]
        )
        frame[f"sequential_greedy_model_rank_{suffix}"] = _rank_descending(
            frame[f"sequential_greedy_{suffix}"]
        )
        frame[f"hungarian_model_rank_{suffix}"] = _rank_descending(
            frame[f"hungarian_{suffix}"]
        )
        frame[f"rank_shift_independent_minus_hungarian_{suffix}"] = (
            frame[f"independent_model_rank_{suffix}"]
            - frame[f"hungarian_model_rank_{suffix}"]
        )
        frame[f"rank_shift_sequential_greedy_minus_hungarian_{suffix}"] = (
            frame[f"sequential_greedy_model_rank_{suffix}"]
            - frame[f"hungarian_model_rank_{suffix}"]
        )
    return frame


def _safe_spearman(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
    statistic = spearmanr(x, y, nan_policy="omit").statistic
    return float(statistic)


def _bootstrap_mean_ci(
    values: np.ndarray, *, n_bootstrap: int, seed: int
) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        means[index] = rng.choice(values, size=len(values), replace=True).mean()
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "estimate": float(values.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "n_models": int(len(values)),
        "n_bootstrap": int(n_bootstrap),
    }


def _top_quartile_overlap(
    frame: pd.DataFrame, left_column: str, right_column: str
) -> dict[str, float | int]:
    n_top = max(1, int(np.ceil(len(frame) * 0.25)))
    left = set(frame.nlargest(n_top, left_column)["model"])
    right = set(frame.nlargest(n_top, right_column)["model"])
    intersection = len(left.intersection(right))
    union = len(left.union(right))
    return {
        "n_top": n_top,
        "n_shared": intersection,
        "fraction_of_quartile_shared": float(intersection / n_top),
        "jaccard": float(intersection / union),
    }


def _agreement_summary(
    frame: pd.DataFrame,
    *,
    left: str,
    right: str,
    delta: str,
) -> dict[str, float]:
    return {
        "spearman_raw": _safe_spearman(
            frame[f"{left}_raw"], frame[f"{right}_raw"]
        ),
        "spearman_adjusted": _safe_spearman(
            frame[f"{left}_adjusted"], frame[f"{right}_adjusted"]
        ),
        "mean_delta_raw": float(frame[f"{delta}_raw"].mean()),
        "median_delta_raw": float(frame[f"{delta}_raw"].median()),
        "mean_delta_adjusted": float(frame[f"{delta}_adjusted"].mean()),
        "median_delta_adjusted": float(frame[f"{delta}_adjusted"].median()),
    }


def _model_agreement_summary(
    frame: pd.DataFrame,
    *,
    left: str,
    right: str,
    delta: str,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = _agreement_summary(
        frame,
        left=left,
        right=right,
        delta=delta,
    )
    summary.update(
        {
            "n_adjusted_delta_below_zero": int(
                (frame[f"{delta}_adjusted"] < 0.0).sum()
            ),
            "top_quartile_overlap_raw": _top_quartile_overlap(
                frame, f"{left}_raw", f"{right}_raw"
            ),
            "top_quartile_overlap_adjusted": _top_quartile_overlap(
                frame, f"{left}_adjusted", f"{right}_adjusted"
            ),
            "mean_adjusted_delta_bootstrap_ci": _bootstrap_mean_ci(
                frame[f"{delta}_adjusted"].to_numpy(),
                n_bootstrap=n_bootstrap,
                seed=seed,
            ),
        }
    )
    return summary


def _compute_statistics(
    *,
    per_dimension: pd.DataFrame,
    per_model: pd.DataFrame,
    pairs: pd.DataFrame,
    selection_distribution: pd.DataFrame,
    independent_raw: np.ndarray,
    sequential_greedy_raw: np.ndarray,
    hungarian_raw: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    independent_minus_hungarian = independent_raw - hungarian_raw
    finite_independent_delta = independent_minus_hungarian[
        np.isfinite(independent_minus_hungarian)
    ]
    hungarian_minus_sequential_pair_mean = np.nansum(
        hungarian_raw - sequential_greedy_raw,
        axis=2,
    ) / hungarian_raw.shape[2]
    off_diagonal = ~np.eye(hungarian_raw.shape[0], dtype=bool)
    finite_hungarian_delta = hungarian_minus_sequential_pair_mean[off_diagonal]
    unused = pairs["unused_source_fraction"]
    zero_row = selection_distribution[
        selection_distribution["selection_count"] == 0
    ]
    zero_fraction = float(zero_row["fraction"].iloc[0]) if len(zero_row) else 0.0
    statistics: dict[str, Any] = {
        "sample_sizes": {
            "models": int(len(per_model)),
            "dimensions": int(len(per_dimension)),
            "ordered_model_pairs": int(len(pairs)),
        },
        "comparisons": {
            "independent_vs_hungarian": {
                "per_dimension": _agreement_summary(
                    per_dimension,
                    left="independent",
                    right="hungarian",
                    delta="delta_independent_minus_hungarian",
                ),
                "per_model": _model_agreement_summary(
                    per_model,
                    left="independent",
                    right="hungarian",
                    delta="delta_independent_minus_hungarian",
                    n_bootstrap=n_bootstrap,
                    seed=seed + 10_000,
                ),
            },
            "sequential_greedy_vs_hungarian": {
                "per_dimension": _agreement_summary(
                    per_dimension,
                    left="sequential_greedy",
                    right="hungarian",
                    delta="delta_sequential_greedy_minus_hungarian",
                ),
                "per_model": _model_agreement_summary(
                    per_model,
                    left="sequential_greedy",
                    right="hungarian",
                    delta="delta_sequential_greedy_minus_hungarian",
                    n_bootstrap=n_bootstrap,
                    seed=seed + 20_000,
                ),
            },
        },
        "independent_best_match_collisions": {
            "mean_unused_source_fraction": float(unused.mean()),
            "median_unused_source_fraction": float(unused.median()),
            "unused_source_fraction_q25": float(unused.quantile(0.25)),
            "unused_source_fraction_q75": float(unused.quantile(0.75)),
            "fraction_of_source_dimensions_never_selected": zero_fraction,
            "mean_target_dimensions_in_collision_fraction": float(
                pairs["target_dimensions_in_collision_fraction"].mean()
            ),
            "median_maximum_claims_on_one_source_dimension": float(
                pairs["maximum_claims_on_one_source_dimension"].median()
            ),
        },
        "invariants": {
            "raw_independent_below_hungarian_dimension_count_tolerance_1e-7": int(
                np.sum(finite_independent_delta < -1e-7)
            ),
            "minimum_raw_independent_minus_hungarian_dimension_score": float(
                finite_independent_delta.min()
            ),
            "raw_hungarian_below_sequential_pair_mean_count_tolerance_1e-7": int(
                np.sum(finite_hungarian_delta < -1e-7)
            ),
            "minimum_raw_hungarian_minus_sequential_pair_mean": float(
                finite_hungarian_delta.min()
            ),
        },
    }
    return statistics


def _canonical_hungarian_validation(
    per_dimension: pd.DataFrame, *, rank: int, dataset: str, n_models: int
) -> dict[str, Any]:
    """Compare raw Hungarian estimates with the saved paper analysis.

    The saved adjusted table may have been produced with the superseded shared
    null shortcut, so it is deliberately not treated as a validation target.
    Raw scores are independent of null calibration and should agree to
    floating-point precision.
    """
    canonical_path = Path(
        f"experiments/universality/shared/data/universality_{dataset}_rank_{rank}.csv"
    )
    if n_models != 162:
        return {
            "performed": False,
            "reason": f"requires all 162 paper models; this run has {n_models}",
            "canonical_csv": str(canonical_path),
        }
    if not canonical_path.exists():
        return {
            "performed": False,
            "reason": "canonical universality CSV is missing",
            "canonical_csv": str(canonical_path),
        }

    canonical = pd.read_csv(canonical_path)
    raw = canonical[canonical["metric"] == "cos2_perm"][
        ["model", "dimension", "universality"]
    ].rename(columns={"universality": "canonical_hungarian_raw"})
    merged = per_dimension.merge(
        raw, on=["model", "dimension"], how="inner", validate="one_to_one"
    )
    expected_rows = n_models * rank
    if len(merged) != expected_rows:
        return {
            "performed": False,
            "reason": f"merged {len(merged)} of {expected_rows} expected rows",
            "canonical_csv": str(canonical_path),
        }
    raw_difference = (
        merged["hungarian_raw"] - merged["canonical_hungarian_raw"]
    ).abs()
    return {
        "performed": True,
        "canonical_csv": str(canonical_path),
        "canonical_csv_sha256": _sha256(canonical_path),
        "n_rows": len(merged),
        "raw_spearman": _safe_spearman(
            merged["hungarian_raw"], merged["canonical_hungarian_raw"]
        ),
        "raw_mae": float(raw_difference.mean()),
        "raw_max_absolute_error": float(raw_difference.max()),
        "note": (
            "Adjusted values are intentionally not compared: saved adjusted "
            "tables generated before pair-specific null calibration are stale "
            "validation targets and must be recomputed."
        ),
    }


def _render_report(
    *,
    params: dict[str, Any],
    statistics: dict[str, Any],
    broad_dimensions: pd.DataFrame,
) -> str:
    independent = statistics["comparisons"]["independent_vs_hungarian"]
    sequential = statistics["comparisons"]["sequential_greedy_vs_hungarian"]
    i_dims = independent["per_dimension"]
    i_models = independent["per_model"]
    s_dims = sequential["per_dimension"]
    s_models = sequential["per_model"]
    collisions = statistics["independent_best_match_collisions"]
    validation = statistics["canonical_hungarian_validation"]
    i_bootstrap = i_models["mean_adjusted_delta_bootstrap_ci"]
    s_bootstrap = s_models["mean_adjusted_delta_bootstrap_ci"]
    top = broad_dimensions.head(5)
    top_lines = "\n".join(
        f"- `{row.source_model}`, dimension {int(row.source_dimension)}: "
        f"selected by {row.target_model_coverage:.1%} of target models "
        f"({int(row.total_claims)} total claims)"
        for row in top.itertuples()
    )
    run_kind = "FULL REVIEW EXPERIMENT"
    if validation["performed"]:
        validation_text = (
            "The Hungarian implementation was also checked against the saved "
            "paper table: raw-score Spearman "
            f"**{validation['raw_spearman']:.6f}** with MAE "
            f"**{validation['raw_mae']:.2e}**. Adjusted values are not compared "
            "because tables produced before pair-specific null calibration are "
            "stale and must be recomputed."
        )
    else:
        validation_text = (
            "Canonical-table validation was not run for this reduced result "
            f"({validation['reason']})."
        )
    return f"""# Matching-rule sensitivity

**{run_kind}.** This run contains {params['n_models']} models, rank
{params['rank']}, and {params['n_perms']} null permutations for every ordered
source–target model pair.

## Result

The phrase *greedy matching* is ambiguous. We therefore distinguish:

1. **Independent best-match (many-to-one):** each target dimension selects its
   best source dimension, with reuse allowed.
2. **Sequential greedy one-to-one:** select the highest-scoring remaining pair,
   remove its target and source dimensions, and repeat.
3. **Hungarian one-to-one:** globally maximize the sum of matched similarities.

Across model dimensions, independent best-match and Hungarian universality have
Spearman correlations of **{i_dims['spearman_raw']:.4f}** before null adjustment
and **{i_dims['spearman_adjusted']:.4f}** after method-specific null adjustment.
Across model means, the adjusted-score correlation is
**{i_models['spearman_adjusted']:.4f}**.

Sequential greedy and Hungarian universality have dimension-level Spearman
correlations of **{s_dims['spearman_raw']:.4f}** before adjustment and
**{s_dims['spearman_adjusted']:.4f}** after adjustment. Across model means, the
adjusted-score correlation is **{s_models['spearman_adjusted']:.4f}**.

At the model level, independent best-match minus Hungarian adjusted
universality is **{i_bootstrap['estimate']:+.4f}** (95% model-bootstrap CI
[{i_bootstrap['ci95_low']:+.4f}, {i_bootstrap['ci95_high']:+.4f}]).
Sequential greedy minus Hungarian is **{s_bootstrap['estimate']:+.4f}**
([{s_bootstrap['ci95_low']:+.4f}, {s_bootstrap['ci95_high']:+.4f}]).
Independent best-match and Hungarian share
**{i_models['top_quartile_overlap_adjusted']['n_shared']} of
{i_models['top_quartile_overlap_adjusted']['n_top']}** top-quartile models;
sequential greedy and Hungarian share
**{s_models['top_quartile_overlap_adjusted']['n_shared']} of
{s_models['top_quartile_overlap_adjusted']['n_top']}**.

For raw pairwise means, independent best-match weakly dominates Hungarian,
which weakly dominates sequential greedy, by construction. Method-specific null
adjustment can change score differences but not the central rank-order
comparison above.

Independent best-match leaves a median
**{collisions['median_unused_source_fraction']:.1%}**
of source dimensions unused per ordered model pair. This quantity is
`1 - n_unique_selected / rank`; it is not the fraction of target dimensions
participating in a duplicate.

The paper's earlier 41% supplemental diagnostic is not a direct benchmark for
this number: that legacy provenance used independent best-match with absolute
Pearson correlation. This experiment uses squared cosine for all three methods,
so it isolates the assignment rule and a different unused fraction is expected.

{validation_text}

## Most broadly selected source dimensions under independent best-match

{top_lines}

## What this does and doesn't show

This comparison changes only the matching rule. Everything else stays fixed:
the selected THINGS factors, rank, model set, squared-cosine similarity,
averaging direction, and permutation procedure. It does not tell us whether
the same result holds at other ranks or on other image datasets. Each matching
rule gets its own null threshold, computed from exactly the same
shuffled similarity matrices.

## Artifacts

- `matching_comparison.pdf` and `.png`: four-panel paper figure.
- `per_dimension.csv`: primary universality estimates and rank shifts.
- `per_model.csv`: model means, ranks, and collision diagnostics.
- `pair_summary.csv`: all ordered target/source model pairs.
- `broad_dimensions.csv`: independent-best-match source selection breadth.
- `selection_count_distribution.csv`: independent-best-match selection counts.
- `pairwise_scores.npz`: raw/adjusted score tensors and assignments.
- `null_thresholds.npz`: method- and ordered-pair-specific target-dimension thresholds.
- `stats.json` and `params.json`: statistics and full run provenance.
"""


def _self_test() -> None:
    rng = np.random.default_rng(7)
    target = rng.normal(size=(80, 6)).astype(np.float32)
    source = rng.normal(size=(80, 6)).astype(np.float32)
    target /= np.maximum(np.linalg.norm(target, axis=0, keepdims=True), 1e-8)
    source /= np.maximum(np.linalg.norm(source, axis=0, keepdims=True), 1e-8)
    (
        independent,
        sequential,
        hungarian,
        independent_assignment,
        sequential_assignment,
    ) = _score_pair(target, source)
    if not np.all(independent + 1e-7 >= hungarian):
        raise AssertionError(
            "Independent best-match must weakly dominate Hungarian per dimension"
        )
    if independent_assignment.shape != (6,) or np.any(
        (independent_assignment < 0) | (independent_assignment >= 6)
    ):
        raise AssertionError("Invalid independent-best-match assignment")
    if not np.array_equal(np.sort(sequential_assignment), np.arange(6)):
        raise AssertionError("Sequential greedy assignment is not one-to-one")
    if hungarian.mean() + 1e-7 < sequential.mean():
        raise AssertionError(
            "Hungarian must weakly dominate sequential greedy in total score"
        )
    cosine_squared = (target.T @ source) ** 2
    rows, columns = linear_sum_assignment(-cosine_squared)
    if len(np.unique(columns)) != 6 or not np.allclose(
        hungarian[rows], cosine_squared[rows, columns]
    ):
        raise AssertionError("Hungarian assignment is not one-to-one")
    if not np.allclose(
        sequential,
        cosine_squared[np.arange(6), sequential_assignment],
    ):
        raise AssertionError("Sequential greedy scores do not match its assignment")
    logger.info("Synthetic three-method matching self-test passed")


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    _self_test()

    if cfg.factorization.rank <= 0:
        raise ValueError("rank must be positive")

    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)
    started_at = datetime.now(timezone.utc)
    started_monotonic = time.monotonic()

    model_names, factor_records, selection_path = _factor_inventory(
        Path(cfg.data_root), cfg.dataset, cfg.factorization.rank, list(cfg.models)
    )
    if (
        not cfg.models
        and cfg.factorization.rank == 50
        and len(model_names) != 162
    ):
        raise RuntimeError(
            "A full paper run requires exactly 162 canonical models, but "
            f"resolved {len(model_names)}"
        )
    signature = _run_signature(
        cfg.factorization.rank, model_names, factor_records, selection_path
    )
    cache_dir = out / "_cache" / signature
    normalized_path, factor_shape = _prepare_normalized_factors(
        rank=cfg.factorization.rank,
        dataset=cfg.dataset,
        model_names=model_names,
        cache_dir=cache_dir,
        signature=signature,
    )
    n_models, n_items, rank = factor_shape
    logger.info(
        "Run %s: %d models × %d items × rank %d",
        signature,
        n_models,
        n_items,
        rank,
    )

    raw_dir = cache_dir / "raw"
    _compute_raw_scores(
        normalized_path=normalized_path,
        raw_dir=raw_dir,
        n_models=n_models,
        rank=rank,
        n_jobs=cfg.n_jobs,
    )
    (
        independent_raw,
        sequential_greedy_raw,
        hungarian_raw,
        independent_assignment,
        sequential_greedy_assignment,
    ) = _assemble_raw(raw_dir, n_models, rank)

    null_dir = cache_dir / f"null_pairwise_p{N_PERMS}_seed{cfg.seed}"
    (
        independent_threshold,
        sequential_greedy_threshold,
        hungarian_threshold,
        _sources_by_target,
    ) = _compute_null_thresholds(
        normalized_path=normalized_path,
        null_dir=null_dir,
        n_models=n_models,
        rank=rank,
        n_perms=N_PERMS,
        seed=cfg.seed,
        n_jobs=cfg.n_jobs,
    )
    independent_adjusted = _null_adjust(independent_raw, independent_threshold)
    sequential_greedy_adjusted = _null_adjust(
        sequential_greedy_raw, sequential_greedy_threshold
    )
    hungarian_adjusted = _null_adjust(hungarian_raw, hungarian_threshold)
    counts = _selection_counts(independent_assignment, rank)

    per_dimension = _per_dimension_table(
        model_names=model_names,
        rank=rank,
        independent_raw=independent_raw,
        sequential_greedy_raw=sequential_greedy_raw,
        hungarian_raw=hungarian_raw,
        independent_adjusted=independent_adjusted,
        sequential_greedy_adjusted=sequential_greedy_adjusted,
        hungarian_adjusted=hungarian_adjusted,
    )
    pairs = _pair_table(
        model_names=model_names,
        rank=rank,
        independent_raw=independent_raw,
        sequential_greedy_raw=sequential_greedy_raw,
        hungarian_raw=hungarian_raw,
        independent_adjusted=independent_adjusted,
        sequential_greedy_adjusted=sequential_greedy_adjusted,
        hungarian_adjusted=hungarian_adjusted,
        selection_counts=counts,
    )
    broad_dimensions = _broad_dimension_table(model_names, counts)
    selection_distribution = _selection_distribution(counts)
    per_model = _per_model_table(
        model_names=model_names,
        per_dimension=per_dimension,
        pairs=pairs,
        broad_dimensions=broad_dimensions,
    )
    statistics = _compute_statistics(
        per_dimension=per_dimension,
        per_model=per_model,
        pairs=pairs,
        selection_distribution=selection_distribution,
        independent_raw=independent_raw,
        sequential_greedy_raw=sequential_greedy_raw,
        hungarian_raw=hungarian_raw,
        n_bootstrap=BOOTSTRAP,
        seed=cfg.seed,
    )
    statistics["canonical_hungarian_validation"] = _canonical_hungarian_validation(
        per_dimension, rank=rank, dataset=cfg.dataset, n_models=n_models
    )

    elapsed = time.monotonic() - started_monotonic
    params: dict[str, Any] = {
        "schema_version": 3,
        "analysis": "matching_rule_sensitivity",
        "dataset": cfg.dataset,
        "rank": rank,
        "n_models": n_models,
        "n_items": n_items,
        "n_ordered_pairs": n_models * (n_models - 1),
        "method": {
            "similarity": "squared cosine",
            "independent_best_match": (
                "independent row maximum; many-to-one source reuse allowed"
            ),
            "sequential_greedy": (
                "highest-scoring remaining pair followed by removal of both "
                "dimensions; one-to-one"
            ),
            "hungarian": "maximum-sum one-to-one linear assignment",
            "aggregation": "mean over every other reference model",
            "null": (
                "95th percentile after permuting source item rows; threshold "
                "computed separately for every ordered source-target pair; "
                "same shuffled cosine-squared matrix used for all methods"
            ),
            "null_adjustment": "clip((raw - p95) / (1 - p95), 0, 1)",
        },
        "n_perms": N_PERMS,
        "n_null_pairs": n_models * (n_models - 1),
        "bootstrap": BOOTSTRAP,
        "seed": cfg.seed,
        "n_jobs": cfg.n_jobs,
        "run_signature": signature,
        "model_names": model_names,
        "provenance": {
            "git_commit": _git_commit(),
            "experiment_source_sha256": {
                path.name: _sha256(path)
                for path in (Path(__file__), Path(__file__).with_name("plot.py"))
            },
            "selection_csv": str(selection_path),
            "selection_sha256": _sha256(selection_path),
            "model_metadata": str(
                Path(cfg.data_root) / "meta" / cfg.dataset / "model_metadata.csv"
            ),
            "model_metadata_sha256": _sha256(
                Path(cfg.data_root) / "meta" / cfg.dataset / "model_metadata.csv"
            ),
            "shared_core": "experiments/universality/shared/core.py",
            "shared_core_sha256": _sha256(
                Path("experiments/universality/shared/core.py")
            ),
            "factor_manifest_sha256": hashlib.sha256(
                json.dumps(
                    factor_records, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "command": [Path(sys.argv[0]).name, *sys.argv[1:]],
        },
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
    }

    _atomic_write_csv(out / "per_dimension.csv", per_dimension)
    _atomic_write_csv(out / "per_model.csv", per_model)
    _atomic_write_csv(out / "pair_summary.csv", pairs)
    _atomic_write_csv(out / "broad_dimensions.csv", broad_dimensions)
    _atomic_write_csv(
        out / "selection_count_distribution.csv",
        selection_distribution,
    )
    _atomic_savez(
        out / "pairwise_scores.npz",
        model_names=np.asarray(model_names),
        independent_raw=independent_raw,
        sequential_greedy_raw=sequential_greedy_raw,
        hungarian_raw=hungarian_raw,
        independent_adjusted=independent_adjusted,
        sequential_greedy_adjusted=sequential_greedy_adjusted,
        hungarian_adjusted=hungarian_adjusted,
        independent_assignment=independent_assignment,
        sequential_greedy_assignment=sequential_greedy_assignment,
        independent_selection_counts=counts,
        # Backward-compatible aliases used by the existing hub analysis.
        greedy_raw=independent_raw,
        greedy_adjusted=independent_adjusted,
        greedy_assignment=independent_assignment,
        greedy_selection_counts=counts,
    )
    _atomic_savez(
        out / "null_thresholds.npz",
        model_names=np.asarray(model_names),
        independent_p95=independent_threshold,
        sequential_greedy_p95=sequential_greedy_threshold,
        hungarian_p95=hungarian_threshold,
        # Backward-compatible alias used by the existing hub analysis.
        greedy_p95=independent_threshold,
    )
    _atomic_write_json(out / "stats.json", statistics)
    _atomic_write_json(out / "params.json", params)
    _atomic_write_text(
        out / "REPORT.md",
        _render_report(
            params=params,
            statistics=statistics,
            broad_dimensions=broad_dimensions,
        ),
    )

    make_figure(
        per_dimension=per_dimension,
        per_model=per_model,
        selection_distribution=selection_distribution,
        statistics=statistics,
        params=params,
        output_dir=out,
    )

    artifact_names = [
        "per_dimension.csv",
        "per_model.csv",
        "pair_summary.csv",
        "broad_dimensions.csv",
        "selection_count_distribution.csv",
        "pairwise_scores.npz",
        "null_thresholds.npz",
        "stats.json",
        "params.json",
        "REPORT.md",
        "matching_comparison.pdf",
        "matching_comparison.png",
    ]
    _atomic_write_json(
        out / "run_complete.json",
        {
            "schema_version": 3,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_signature": signature,
            "artifacts": {
                name: {
                    "bytes": (out / name).stat().st_size,
                    "sha256": _sha256(out / name),
                }
                for name in artifact_names
            },
        },
    )
    logger.info("Complete in %.1f seconds; outputs: %s", elapsed, out)


if __name__ == "__main__":
    main()
