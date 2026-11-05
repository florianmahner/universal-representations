#!/usr/bin/env python3
"""Turn the raw participant-level rating exports into per-trial tables."""
from __future__ import annotations

import csv
import html
import json
import logging
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

EXPECTED_PARTICIPANTS = 570
EXPECTED_VERSIONS = 71

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def build_llm_content_map(tasks_path: Path) -> dict[str, str]:
    """Build dim_id → llm_content mapping from tasks.csv."""
    mapping: dict[str, str] = {}
    with tasks_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = html.unescape(row["trials_json"])
            trials = json.loads(raw)
            for t in trials:
                dim_id = t["dim_id"]
                if dim_id not in mapping:
                    mapping[dim_id] = t.get("content", "")
    log.info(f"  llm_content map: {len(mapping)} dim_ids")
    return mapping


def parse_submission(participant: str, task_data: dict) -> list[dict]:
    """Extract per-trial rows from a single participant's taskData dict."""
    version = task_data.get("version", "")
    n_trials = int(task_data.get("n_trials", 18))

    rows = []
    for i in range(1, n_trials + 1):
        dim_id = task_data.get(f"trial{i}_dim_id")
        task_type = task_data.get(f"trial{i}_task_type")
        category = task_data.get(f"trial{i}_category", "")
        interpretability_raw = task_data.get(f"trial{i}_interpretability", "")
        time_ms_raw = task_data.get(f"trial{i}_time_ms", "")

        if not dim_id or not task_type:
            continue

        try:
            interpretability = int(interpretability_raw) if interpretability_raw else None
        except ValueError:
            interpretability = None

        try:
            time_ms = float(time_ms_raw) if time_ms_raw else None
        except ValueError:
            time_ms = None

        rows.append({
            "participant": participant,
            "version": version,
            "dim_id": dim_id,
            "task_type": task_type,
            "category": category,
            "interpretability": interpretability,
            "time_ms": time_ms,
        })
    return rows


def parse_assignment_file(path: Path, llm_map: dict[str, str]) -> list[dict]:
    """Parse one assignments CSV file, return list of trial rows."""
    rows = []
    n_skipped = 0
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("CompletionType") != "Template":
                n_skipped += 1
                continue
            submitted_raw = row.get("Submitted Data", "")
            if not submitted_raw:
                n_skipped += 1
                continue
            try:
                submitted = json.loads(submitted_raw)
                task_data = submitted["Data"]["taskData"]
            except (json.JSONDecodeError, KeyError):
                n_skipped += 1
                continue

            participant = row["ParticipantId"]
            trial_rows = parse_submission(participant, task_data)

            # Attach llm_content and model
            for r in trial_rows:
                r["llm_content"] = llm_map.get(r["dim_id"], "")
                # model is the part before __d
                if "__d" in r["dim_id"]:
                    r["model"] = r["dim_id"].rsplit("__d", 1)[0]
                else:
                    r["model"] = ""

            rows.extend(trial_rows)

    log.info(f"  {path.name}: {len(rows)} trial rows ({n_skipped} rows skipped)")
    return rows


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)

    # The participant-level exports are not bundled; drop the shards of the
    # single 570-participant cohort reported in the manuscript here, and do not
    # mix in independent collection waves.
    assignment_files = sorted((Path(cfg.data_root) / "ratings").glob("assignments_*.csv"))
    if not assignment_files:
        raise FileNotFoundError(
            f"No assignments_*.csv in {Path(cfg.data_root) / 'ratings'}"
        )
    tasks = out / "tasks.csv"
    if not tasks.is_file():
        raise FileNotFoundError(f"Task file not found: {tasks}")

    log.info(f"Building llm_content map from {tasks}...")
    llm_map = build_llm_content_map(tasks)

    log.info(f"\nParsing {len(assignment_files)} cohort file(s):")
    for p in assignment_files:
        log.info(f"  {p.name}")

    all_rows: list[dict] = []
    for path in assignment_files:
        all_rows.extend(parse_assignment_file(path, llm_map))

    if not all_rows:
        log.error("No rows parsed — check assignment files.")
        return

    df = pd.DataFrame(all_rows)

    n_participants = df["participant"].nunique()
    n_versions = df["version"].nunique()
    if n_participants != EXPECTED_PARTICIPANTS:
        raise ValueError(
            "Selected exports do not match the paper cohort: expected "
            f"{EXPECTED_PARTICIPANTS} participants, found {n_participants}. "
            "Do not combine independent collection waves."
        )
    if n_versions != EXPECTED_VERSIONS:
        raise ValueError(
            "Selected exports do not match the paper design: expected "
            f"{EXPECTED_VERSIONS} survey versions, found {n_versions}."
        )

    # Remove accidental export duplication. The intentional repeat trial has a
    # distinct R_ dim_id and is therefore preserved.
    n_before = len(df)
    df = df.drop_duplicates(subset=["participant", "dim_id"])
    if len(df) < n_before:
        log.info(f"\nDropped {n_before - len(df)} duplicate participant×dim_id rows")

    # Compute time_s
    df["time_s"] = df["time_ms"] / 1000.0

    # Column order matching existing parsed_all_trials.csv
    col_order = [
        "participant", "version", "dim_id", "model",
        "task_type", "category", "interpretability",
        "time_ms", "llm_content", "time_s",
    ]
    df = df[col_order]

    out_all = out / "parsed_all_trials.csv"
    out_real = out / "parsed_real_trials.csv"
    df.to_csv(out_all, index=False)
    log.info(f"\nWrote {len(df)} rows → {out_all}")

    real = df[df["task_type"] == "real"].copy()
    real.to_csv(out_real, index=False)
    log.info(f"Wrote {len(real)} real-trial rows → {out_real}")

    log.info(f"\nParticipants:   {df['participant'].nunique()}")
    log.info(f"Dimensions:     {df[df['task_type'] == 'real']['dim_id'].nunique()}")
    log.info(f"Task type breakdown:")
    for tt, n in df["task_type"].value_counts().items():
        log.info(f"  {tt}: {n}")


if __name__ == "__main__":
    main()
