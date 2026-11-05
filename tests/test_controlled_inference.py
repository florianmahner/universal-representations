import numpy as np
import pandas as pd

from experiments.universality.controlled_comparisons.contrasts.uniform_tests.run import (
    apply_corrections,
    run_clip_vit_convergence,
    run_contrast,
)


def _frame(groups):
    return pd.DataFrame(
        {
            "comparison": [name for name, values in groups for _ in values],
            "universality": [
                value for _name, values in groups for value in values
            ],
        }
    )


def test_two_group_contrast_uses_welch_and_hedges_g():
    frame = _frame([("a", [0.1, 0.2, 0.3]), ("b", [0.4, 0.5, 0.7])])

    result = run_contrast(frame, "comparison", n_bootstrap=100, seed=3)

    assert result["test"] == "Welch t"
    assert result["effect_size_type"] == "hedges_g"
    assert np.isfinite(result["effect_ci_low"])
    assert np.isfinite(result["effect_ci_high"])


def test_nominal_multigroup_contrast_stays_anova_with_singleton_group():
    frame = _frame(
        [
            ("a", [0.1, 0.2, 0.3]),
            ("b", [0.4, 0.5, 0.7]),
            ("singleton", [0.9]),
        ]
    )

    result = run_contrast(frame, "comparison", n_bootstrap=100, seed=4)

    assert result["n_groups"] == 3
    assert result["n_groups_testable"] == 2
    assert result["test"] == "one-way ANOVA"
    assert result["effect_size_type"] == "partial_eta_squared"
    assert np.isfinite(result["omega_squared"])
    assert np.isfinite(result["omega_ci_low"])
    assert np.isfinite(result["omega_ci_high"])


def test_bonferroni_is_the_primary_familywise_correction():
    table = pd.DataFrame({"p_value": [0.01, 0.03, np.nan]})

    corrected = apply_corrections(table)

    np.testing.assert_allclose(
        corrected.loc[:1, "p_bonferroni"].to_numpy(), [0.03, 0.09]
    )
    np.testing.assert_allclose(
        corrected.loc[:1, "p_bonferroni_all_prespecified"].to_numpy(),
        [0.03, 0.09],
    )
    np.testing.assert_allclose(
        corrected.loc[:1, "p_bonferroni_testable"].to_numpy(),
        [0.02, 0.06],
    )
    assert corrected["n_tests_in_family"].eq(3).all()
    assert corrected["n_testable_contrasts"].eq(2).all()
    assert corrected["n_prespecified_contrasts"].eq(3).all()
    assert corrected.loc[0, "sig_bonferroni"]
    assert not corrected.loc[1, "sig_bonferroni"]


def test_clip_vit_convergence_uses_size_matched_without_replacement_null():
    frame = pd.DataFrame(
        {
            "universality": np.linspace(0.0, 1.0, 20),
            "compare_diet_vit": [f"dataset_{i}" if i < 7 else np.nan
                                 for i in range(20)],
        }
    )

    result = run_clip_vit_convergence(frame, n_bootstrap=100, seed=7)

    assert result["n_group"] == 7
    assert result["null_pool_size"] == 20
    assert result["sampling"] == "without replacement"
    assert result["n_resamples"] == 100
    assert 0.0 < result["p_value_lower_tail"] <= 1.0
