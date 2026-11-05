import pandas as pd

from experiments.human_ratings.run import (
    compute_split_half,
    quality_check,
)


def test_quality_control_excludes_only_attention_failures():
    trials = pd.DataFrame(
        [
            {
                "participant": "passes_attention",
                "task_type": "attention_check",
                "category": "both",
                "interpretability": 4,
            },
            {
                "participant": "passes_attention",
                "task_type": "catch",
                "category": "neither",
                "interpretability": 2,
            },
            {
                "participant": "passes_attention",
                "task_type": "real",
                "category": "visual",
                "interpretability": 3,
            },
            {
                "participant": "fails_attention",
                "task_type": "attention_check",
                "category": "semantic",
                "interpretability": 4,
            },
            {
                "participant": "fails_attention",
                "task_type": "catch",
                "category": "semantic",
                "interpretability": 2,
            },
            {
                "participant": "fails_attention",
                "task_type": "real",
                "category": "semantic",
                "interpretability": 3,
            },
        ]
    )

    clean, qc = quality_check(trials)

    assert set(clean["participant"]) == {"passes_attention"}
    indexed = qc.set_index("participant")
    assert bool(indexed.loc["passes_attention", "pass"])
    assert not bool(indexed.loc["passes_attention", "catch_pass"])
    assert not bool(indexed.loc["fails_attention", "pass"])


def test_split_half_enumerates_every_unique_four_plus_four_partition():
    real = pd.DataFrame(
        {
            "dim_id": ["model__d0"] * 8,
            "category": ["semantic"] * 8,
        }
    )

    result = compute_split_half(real)

    assert len(result) == 1
    assert result.loc[0, "n_partitions"] == 35
    assert result.loc[0, "split_half"] == 1.0
