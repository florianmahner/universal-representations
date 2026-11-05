from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load as joblib_load
from PIL import Image


class ThingsDataset:
    """
    Unified data loader for THINGS images, model features, and NMF components.

    Parameters
    ----------
    data_root : Path | str
        Root of the data tree, holding ``features/``, ``embeddings/``, ``meta/``
        and ``images/``.  Relative paths resolve against the repository root.
    dataset : str
        Name of the feature set below ``features/``, ``embeddings/`` and ``meta/``.
    """

    def __init__(
        self,
        data_root: Path | str = Path("data"),
        dataset: str = "things22k",
    ):
        self.data_root = Path(data_root)
        self.dataset = dataset

        self._image_info: pd.DataFrame | None = None
        self._model_info: pd.DataFrame | None = None
        self._all_image_paths: list[Path] | None = None
        self._all_categories: list[str] | None = None
        self._category_to_indices: dict[str, list[int]] | None = None
        self._exemplary_indices: list[int] | None = None
        self._active_indices: list[int] | None = None
        self._bandwidth_selection_cache: dict[int, pd.DataFrame] = {}

    def _load_image_info(self) -> None:
        if self._image_info is not None:
            return

        image_info_path = (
            self.data_root / "meta" / self.dataset / "image_info_22k.csv"
        )
        if not image_info_path.exists():
            raise FileNotFoundError(f"Image info not found: {image_info_path}")

        self._image_info = pd.read_csv(image_info_path)
        if "category" not in self._image_info.columns:
            raise ValueError(f"Missing 'category' column in {image_info_path}")

        if "path" in self._image_info.columns:
            raw_paths = self._image_info["path"].astype(str).tolist()
        elif {"filename", "category"}.issubset(self._image_info.columns):
            raw_paths = [
                str(Path(cat) / fname)
                for cat, fname in zip(self._image_info["category"], self._image_info["filename"])
            ]
        else:
            raise ValueError(
                f"Expected either 'path' or ('category','filename') columns in {image_info_path}"
            )

        images_root = self.data_root / "images" / "things"
        self._all_image_paths = []
        for raw_path in raw_paths:
            path = Path(raw_path)
            if path.exists():
                self._all_image_paths.append(path)
            else:
                rel_path = Path(path.parent.name) / path.name
                self._all_image_paths.append(images_root / rel_path)

        self._all_categories = self._image_info["category"].astype(str).tolist()

        self._category_to_indices = {}
        for idx, cat in enumerate(self._all_categories):
            self._category_to_indices.setdefault(cat, []).append(idx)

        self._exemplary_indices = []
        for _, indices in sorted(self._category_to_indices.items()):
            pick_idx = 1 if len(indices) > 1 else 0
            self._exemplary_indices.append(indices[pick_idx])

    def _default_indices(self) -> list[int]:
        if self._active_indices is not None:
            return self._active_indices

        self._load_image_info()
        inferred_rows = self._infer_component_row_count()
        n_all = len(self._all_image_paths)
        n_exemplars = len(self._exemplary_indices)

        if inferred_rows == n_all:
            self._active_indices = list(range(n_all))
            return self._active_indices
        if inferred_rows == n_exemplars:
            self._active_indices = list(self._exemplary_indices)
            return self._active_indices

        self._active_indices = list(range(n_all))
        return self._active_indices

    def _infer_component_row_count(self) -> int | None:
        models = self.list_component_models()
        for model in models:
            ranks = self.list_available_ranks(model)
            if not ranks:
                continue
            try:
                w = self.load_components(model, rank=ranks[0])
                return int(w.shape[0])
            except Exception:
                continue
        return None

    @property
    def image_paths(self) -> list[Path]:
        self._load_image_info()
        return [self._all_image_paths[i] for i in self._default_indices()]

    @property
    def categories(self) -> list[str]:
        self._load_image_info()
        return [self._all_categories[i] for i in self._default_indices()]

    @property
    def n_images(self) -> int:
        return len(self._default_indices())

    @property
    def n_categories(self) -> int:
        self._load_image_info()
        return len(self._category_to_indices)

    def get_image(self, idx: int) -> Image.Image:
        self._load_image_info()
        full_idx = self._default_indices()[idx]
        return Image.open(self._all_image_paths[full_idx])

    @property
    def all_image_paths(self) -> list[Path]:
        self._load_image_info()
        return self._all_image_paths

    @property
    def all_categories(self) -> list[str]:
        self._load_image_info()
        return self._all_categories

    @property
    def n_all_images(self) -> int:
        self._load_image_info()
        return len(self._all_image_paths)

    def get_all_image(self, idx: int) -> Image.Image:
        self._load_image_info()
        return Image.open(self._all_image_paths[idx])

    def list_embedding_models(self) -> list[str]:
        feature_dir = self.data_root / "features" / self.dataset
        if not feature_dir.exists():
            return []
        return sorted(f.stem for f in feature_dir.glob("*.npy"))

    def load_embedding(self, model: str, full: bool = False) -> np.ndarray:
        path = self.data_root / "features" / self.dataset / f"{model}.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"Embedding not found: {path}\n"
                f"Available models: {self.list_embedding_models()[:10]}..."
            )

        emb = np.load(path)
        if full:
            return emb
        return emb[self._default_indices()]

    def list_component_models(self) -> list[str]:
        embeddings_dir = self.data_root / "embeddings" / self.dataset
        if not embeddings_dir.exists():
            return []
        models = []
        for model_dir in embeddings_dir.iterdir():
            if not model_dir.is_dir():
                continue
            has_rank = any(
                d.is_dir() and d.name.startswith("rank_")
                for d in model_dir.iterdir()
            )
            if has_rank:
                models.append(model_dir.name)
        return sorted(models)

    def list_available_ranks(self, model: str) -> list[int]:
        model_dir = self.data_root / "embeddings" / self.dataset / model
        if not model_dir.exists():
            return []
        ranks: list[int] = []
        for d in model_dir.iterdir():
            if d.is_dir() and d.name.startswith("rank_"):
                try:
                    ranks.append(int(d.name.split("_", maxsplit=1)[1]))
                except ValueError:
                    continue
        return sorted(ranks)

    def _resolve_rank(self, model: str, rank: int | None) -> int:
        if rank is not None:
            return rank
        available = self.list_available_ranks(model)
        if len(available) == 1:
            return available[0]
        raise ValueError(
            f"rank is required for model '{model}'. Available ranks: {available}"
        )

    def load_components(
        self,
        model: str,
        source: str = "full_fit",
        rank: int | None = None,
    ) -> np.ndarray:
        resolved_rank = self._resolve_rank(model, rank)
        sel = self.bandwidth_selection(resolved_rank)
        row = sel[sel["model"] == model]
        if row.empty:
            raise ValueError(
                f"No sigma selection for model '{model}' at rank {resolved_rank}. "
                f"Run python -m experiments.factorization.bandwidth_selection.compute_selection "
                f"factorization.rank={resolved_rank} first."
            )
        sigma = row.iloc[0]["sigma_mult"]
        seed = int(row.iloc[0]["best_seed"])
        seed_dir = (
            self.data_root / "embeddings" / self.dataset / model / f"rank_{resolved_rank}"
            / f"sigma_{sigma}" / f"seed_{seed}"
        )
        npz_path = seed_dir / "w.npz"
        npy_path = seed_dir / "w.npy"
        if npz_path.exists():
            return np.load(npz_path)["w"]
        if npy_path.exists():
            return np.load(npy_path)
        raise FileNotFoundError(f"Optimal components not found in: {seed_dir}")

    def bandwidth_selection(self, rank: int = 50) -> pd.DataFrame:
        if rank not in self._bandwidth_selection_cache:
            path = (
                self.data_root / "meta" / self.dataset / f"bandwidth_selection_rank_{rank}.csv"
            )
            if not path.exists():
                raise FileNotFoundError(
                    f"bandwidth_selection_rank_{rank}.csv not found at {path}. "
                    f"Run python -m experiments.factorization.bandwidth_selection.compute_selection "
                    f"factorization.rank={rank} first."
                )
            self._bandwidth_selection_cache[rank] = pd.read_csv(path)
        return self._bandwidth_selection_cache[rank]

    def load_component_metadata(
        self,
        model: str,
        source: str = "full_fit",
        rank: int | None = None,
    ) -> dict:
        resolved_rank = self._resolve_rank(model, rank)
        rank_dir = (
            self.data_root / "embeddings" / self.dataset / model / f"rank_{resolved_rank}"
        )
        json_path = rank_dir / "metrics.json"
        joblib_path = rank_dir / "metrics.joblib"
        if json_path.exists():
            return json.loads(json_path.read_text())
        if joblib_path.exists():
            return joblib_load(joblib_path)
        return {}

    @staticmethod
    def _normalize_model_info(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "name" not in df.columns:
            for candidate in ("feature_uid", "model_uid", "model"):
                if candidate in df.columns:
                    df["name"] = df[candidate].astype(str)
                    break

        if "model_uid" not in df.columns and "name" in df.columns:
            df["model_uid"] = df["name"]

        if "penultimate" not in df.columns and "penultimate_layer" in df.columns:
            df["penultimate"] = df["penultimate_layer"]
        if "weights" not in df.columns and "weights_url" in df.columns:
            df["weights"] = df["weights_url"]

        return df

    @property
    def model_info(self) -> pd.DataFrame:
        if self._model_info is None:
            path = self.data_root / "meta" / self.dataset / "model_metadata.csv"
            if not path.exists():
                raise FileNotFoundError(f"Model info not found: {path}")
            raw = pd.read_csv(path, comment="#")
            self._model_info = self._normalize_model_info(raw)
        return self._model_info

    def get_model_info(self, model_name: str) -> pd.Series:
        df = self.model_info
        masks = []
        for col in ("name", "model_uid", "feature_uid"):
            if col in df.columns:
                masks.append(df[col] == model_name)
        if not masks:
            raise ValueError("Model info table has no identifier columns")

        mask = masks[0]
        for m in masks[1:]:
            mask |= m
        matches = df[mask]

        if len(matches) == 0:
            raise ValueError(
                f"Unknown model: {model_name}\n"
                f"Available models: {df['name'].tolist()[:20]}..."
            )
        return matches.iloc[0]

    def __repr__(self) -> str:
        try:
            n_models_with_info = len(self.model_info)
        except Exception:
            n_models_with_info = 0
        return (
            "ThingsDataset(\n"
            f"  data_root={self.data_root},\n"
            f"  dataset={self.dataset},\n"
            f"  n_images={self.n_images} (default indexed),\n"
            f"  n_all_images={self.n_all_images},\n"
            f"  n_categories={self.n_categories},\n"
            f"  n_embedding_models={len(self.list_embedding_models())},\n"
            f"  n_component_models={len(self.list_component_models())},\n"
            f"  n_models_with_info={n_models_with_info},\n"
            ")"
        )
