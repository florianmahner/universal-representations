from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import OmegaConf
from scipy import stats

from src.colors import CYCLE, GRAY, GRAY_DARK, GRAY_PALE, INDIGO
from src.plotting import create_figure, save_figure, setup_plotting


def prepare_regression_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare a clean DataFrame for regression with encoded categoricals."""
    reg = df[["model", "universality"]].copy()

    # Continuous predictors
    if "top1_accuracy" in df.columns:
        reg["accuracy"] = df["top1_accuracy"]
    if "total_parameter_count" in df.columns:
        reg["log_params"] = np.log10(df["total_parameter_count"].replace(0, np.nan))

    # Categorical predictors
    if "model_class" in df.columns:
        reg["is_transformer"] = (df["model_class"] == "Transformer").astype(float)
        reg["is_cnn"] = (df["model_class"] == "Convolutional").astype(float)

    if "objective_category" in df.columns:
        reg["is_selfsupervised"] = (df["objective_category"] == "Self-Supervised").astype(float)
        reg["is_visionlanguage"] = (df["objective_category"] == "Vision-Language").astype(float)

    # Training data scale proxy
    if "compare_diet_imagenetsize" in df.columns:
        reg["is_imagenet21k"] = (df["compare_diet_imagenetsize"] == "imagenet21k").astype(float)

    # Drop columns that are constant (zero variance) — they break regression
    for col in list(reg.columns):
        if col in ("model", "universality"):
            continue
        if reg[col].dropna().nunique() <= 1:
            reg.drop(columns=[col], inplace=True)

    return reg


def pairwise_correlations(reg: pd.DataFrame) -> pd.DataFrame:
    """Compute correlations between universality and all numeric predictors."""
    numeric_cols = [c for c in reg.columns if c not in ("model", "universality") and reg[c].dtype in ("float64", "float32", "int64")]
    rows = []
    for col in numeric_cols:
        sub = reg[["universality", col]].dropna()
        if len(sub) < 5:
            continue
        r, p = stats.pearsonr(sub["universality"], sub[col])
        rho, p_rho = stats.spearmanr(sub["universality"], sub[col])
        rows.append({
            "predictor": col,
            "n": len(sub),
            "pearson_r": r,
            "pearson_p": p,
            "spearman_rho": rho,
            "spearman_p": p_rho,
        })
    return pd.DataFrame(rows).sort_values("pearson_p")


def multiple_regression(reg: pd.DataFrame) -> dict:
    """OLS multiple regression predicting universality from all available predictors."""
    predictors = [c for c in reg.columns if c not in ("model", "universality")]
    sub = reg[["universality"] + predictors].dropna()
    if len(sub) < 10:
        return {"error": "Too few complete cases"}

    # Drop predictors that are constant in the complete-case subset
    predictors = [p for p in predictors if sub[p].nunique() > 1]
    sub = sub[["universality"] + predictors]

    y = sub["universality"].values
    X = sub[predictors].values
    # Add intercept
    X_int = np.column_stack([np.ones(len(X)), X])

    # OLS via pseudoinverse
    beta = np.linalg.lstsq(X_int, y, rcond=None)[0]
    y_hat = X_int @ beta
    residuals = y - y_hat
    ss_res = (residuals**2).sum()
    ss_tot = ((y - y.mean())**2).sum()
    r_squared = 1 - ss_res / ss_tot
    n, p = X.shape
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)

    # Standard errors (use pseudoinverse for robustness)
    mse = ss_res / (n - p - 1)
    try:
        cov = mse * np.linalg.pinv(X_int.T @ X_int)
        se = np.sqrt(np.abs(np.diag(cov)))
    except np.linalg.LinAlgError:
        se = np.full(p + 1, np.nan)

    t_vals = beta / se
    p_vals = 2 * stats.t.sf(np.abs(t_vals), df=n - p - 1)

    coef_df = pd.DataFrame({
        "predictor": ["intercept"] + predictors,
        "beta": beta,
        "se": se,
        "t": t_vals,
        "p": p_vals,
    })

    return {
        "n": n,
        "n_predictors": p,
        "r_squared": r_squared,
        "adj_r_squared": adj_r_squared,
        "coefficients": coef_df,
        "predictors_used": predictors,
    }


def partial_correlations(reg: pd.DataFrame) -> pd.DataFrame:
    """Compute partial correlations for each predictor.

    Residualizes both universality and the target predictor on all other
    predictors, then correlates the residuals.
    """
    predictors = [c for c in reg.columns if c not in ("model", "universality")]
    sub = reg[["universality"] + predictors].dropna()
    if len(sub) < 10:
        return pd.DataFrame()

    # Drop constant predictors in the complete-case subset
    predictors = [p for p in predictors if sub[p].nunique() > 1]
    sub = sub[["universality"] + predictors]

    y = sub["universality"].values
    rows = []
    for target_pred in predictors:
        other_preds = [p for p in predictors if p != target_pred]
        if not other_preds:
            continue

        X_others = sub[other_preds].values
        X_others_int = np.column_stack([np.ones(len(X_others)), X_others])

        # Residualize y on others
        beta_y = np.linalg.lstsq(X_others_int, y, rcond=None)[0]
        resid_y = y - X_others_int @ beta_y

        # Residualize target on others
        x_target = sub[target_pred].values
        beta_x = np.linalg.lstsq(X_others_int, x_target, rcond=None)[0]
        resid_x = x_target - X_others_int @ beta_x

        r, p = stats.pearsonr(resid_y, resid_x)
        rows.append({
            "predictor": target_pred,
            "partial_r": r,
            "partial_r_squared": r**2,
            "p_value": p,
            "n": len(sub),
        })

    return pd.DataFrame(rows).sort_values("partial_r_squared", ascending=False)


def variance_partitioning(reg: pd.DataFrame) -> pd.DataFrame:
    """Hierarchical regression: compute incremental R² for each predictor category.

    Groups predictors into architecture, training objective, and scale/performance.
    """
    predictor_groups = {
        "Architecture": ["is_transformer", "is_cnn"],
        "Training Objective": ["is_selfsupervised", "is_visionlanguage"],
        "Scale / Performance": ["log_params", "accuracy", "is_imagenet21k"],
    }

    all_preds = []
    for preds in predictor_groups.values():
        all_preds.extend(preds)
    available = [p for p in all_preds if p in reg.columns]
    sub = reg[["universality"] + available].dropna()

    if len(sub) < 10:
        return pd.DataFrame()

    y = sub["universality"].values

    def r2_for(pred_list):
        cols = [p for p in pred_list if p in sub.columns]
        if not cols:
            return 0.0
        X = sub[cols].values
        X_int = np.column_stack([np.ones(len(X)), X])
        beta = np.linalg.lstsq(X_int, y, rcond=None)[0]
        y_hat = X_int @ beta
        ss_res = ((y - y_hat)**2).sum()
        ss_tot = ((y - y.mean())**2).sum()
        return 1 - ss_res / ss_tot

    full_r2 = r2_for(available)

    rows = []
    for group_name, group_preds in predictor_groups.items():
        # R² without this group
        others = [p for p in available if p not in group_preds]
        r2_without = r2_for(others) if others else 0.0
        delta_r2 = full_r2 - r2_without

        # R² of this group alone
        r2_alone = r2_for(group_preds)

        rows.append({
            "predictor_group": group_name,
            "r2_alone": r2_alone,
            "r2_incremental": delta_r2,
            "r2_shared": r2_alone - delta_r2,
        })

    rows.append({
        "predictor_group": "Full Model",
        "r2_alone": full_r2,
        "r2_incremental": full_r2,
        "r2_shared": 0.0,
    })

    return pd.DataFrame(rows)


def plot_correlation_matrix(reg: pd.DataFrame) -> plt.Figure:
    """Heatmap of pairwise correlations among universality and all predictors."""
    numeric = reg.select_dtypes(include=[np.number]).drop(columns=["universality"], errors="ignore")
    cols_to_use = ["universality"] + [c for c in numeric.columns if c != "universality"]
    corr_sub = reg[cols_to_use].dropna()
    if len(corr_sub) < 5:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Too few complete cases", ha="center", va="center")
        return fig

    corr = corr_sub.corr()

    fig, ax = plt.subplots(figsize=(8, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
        center=0, vmin=-1, vmax=1, square=True, ax=ax,
        linewidths=0.5, cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Correlation Matrix: Universality and Model Properties")
    fig.tight_layout()
    return fig


def plot_regression_coefficients(result: dict) -> plt.Figure:
    """Forest plot of standardized regression coefficients."""
    coef_df = result["coefficients"]
    coef_df = coef_df[coef_df["predictor"] != "intercept"].copy()

    fig, ax = plt.subplots(figsize=(7, max(3, len(coef_df) * 0.6)))
    colors = [INDIGO if p < 0.05 else GRAY_PALE for p in coef_df["p"]]
    edge_colors = [GRAY_DARK if p < 0.05 else "gray" for p in coef_df["p"]]

    y_pos = range(len(coef_df))
    ax.barh(y_pos, coef_df["beta"], xerr=coef_df["se"] * 1.96,
            color=colors, edgecolor=edge_colors, linewidth=1.2, capsize=3)

    for i, (_, row) in enumerate(coef_df.iterrows()):
        p_str = f"p={row['p']:.3f}" if row["p"] >= 0.001 else "p<.001"
        ax.text(
            row["beta"] + row["se"] * 2.2 if row["beta"] >= 0 else row["beta"] - row["se"] * 2.2,
            i, f" {p_str}", va="center",
            ha="left" if row["beta"] >= 0 else "right", fontsize=8,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(coef_df["predictor"], fontsize=9)
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Regression Coefficient (unstandardized)")
    ax.set_title(
        f"Multiple Regression: R² = {result['r_squared']:.3f}, "
        f"adj. R² = {result['adj_r_squared']:.3f}, n = {result['n']}"
    )
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def plot_partial_correlations(partial_df: pd.DataFrame) -> plt.Figure:
    """Bar chart of partial correlations (unique variance per predictor)."""
    if partial_df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No partial correlations computed", ha="center", va="center")
        return fig

    partial_df = partial_df.sort_values("partial_r_squared", ascending=True).copy()

    fig, ax = plt.subplots(figsize=(7, max(3, len(partial_df) * 0.6)))
    colors = [INDIGO if p < 0.05 else GRAY_PALE for p in partial_df["p_value"]]
    edge_colors = [GRAY_DARK if p < 0.05 else "gray" for p in partial_df["p_value"]]

    ax.barh(range(len(partial_df)), partial_df["partial_r_squared"],
            color=colors, edgecolor=edge_colors, linewidth=1.2)

    for i, (_, row) in enumerate(partial_df.iterrows()):
        p_str = f"p={row['p_value']:.3f}" if row["p_value"] >= 0.001 else "p<.001"
        ax.text(row["partial_r_squared"] + 0.002, i,
                f" r={row['partial_r']:.3f}, {p_str}", va="center", fontsize=8)

    ax.set_yticks(range(len(partial_df)))
    ax.set_yticklabels(partial_df["predictor"], fontsize=9)
    ax.set_xlabel("Partial r² (unique variance explained)")
    ax.set_title("Partial Correlations: Unique Contribution to Universality")
    fig.tight_layout()
    return fig


def plot_variance_partitioning(vp_df: pd.DataFrame) -> plt.Figure:
    """Stacked bar chart of variance partitioning."""
    if vp_df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No variance partitioning computed", ha="center", va="center")
        return fig

    groups = vp_df[vp_df["predictor_group"] != "Full Model"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: R² alone vs incremental
    x = range(len(groups))
    w = 0.35
    axes[0].bar([i - w/2 for i in x], groups["r2_alone"], w,
                label="R² alone", color=CYCLE[0], edgecolor="white")
    axes[0].bar([i + w/2 for i in x], groups["r2_incremental"], w,
                label="R² incremental (unique)", color=CYCLE[1], edgecolor="white")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(groups["predictor_group"], fontsize=9)
    axes[0].set_ylabel("R²")
    axes[0].set_title("Variance Explained: Alone vs. Unique")
    axes[0].legend(fontsize=8)

    # Right: Stacked bar decomposition
    full_r2 = vp_df.loc[vp_df["predictor_group"] == "Full Model", "r2_alone"].values[0]
    unique_vals = groups["r2_incremental"].values
    shared_vals = groups["r2_shared"].values
    unexplained = 1.0 - full_r2

    labels = list(groups["predictor_group"]) + ["Shared", "Unexplained"]
    sizes = list(unique_vals) + [shared_vals.sum() if shared_vals.sum() > 0 else 0, unexplained]
    sizes = [max(0, s) for s in sizes]  # clamp negatives

    colors_pie = CYCLE[:len(groups)] + [GRAY_PALE, "#f5f5f5"]
    axes[1].pie(sizes, labels=labels, colors=colors_pie, autopct=lambda p: f"{p:.1f}%" if p > 2 else "",
                startangle=90, textprops={"fontsize": 8})
    axes[1].set_title(f"Variance Decomposition (Total R² = {full_r2:.3f})")

    fig.tight_layout()
    return fig


def plot_scatter_continuous(df: pd.DataFrame) -> plt.Figure:
    """Scatter plots of universality vs continuous predictors."""
    cont_cols = []
    if "top1_accuracy" in df.columns and df["top1_accuracy"].notna().sum() > 5:
        cont_cols.append(("top1_accuracy", "Top-1 accuracy (%)"))
    if "total_parameter_count" in df.columns and df["total_parameter_count"].notna().sum() > 5:
        cont_cols.append(("total_parameter_count", "Parameters (log scale)"))

    if not cont_cols:
        fig, ax = create_figure("single")
        ax.text(0.5, 0.5, "No continuous predictors", ha="center", va="center")
        return fig

    fig, axes = create_figure("single", ncols=len(cont_cols))
    if len(cont_cols) == 1:
        axes = [axes]

    for ax, (col, label) in zip(axes, cont_cols):
        sub = df[df[col].notna()].copy()
        x = sub[col].values
        if "parameter" in col.lower():
            x = np.log10(x + 1)

        if "model_class" in sub.columns:
            classes = sorted(sub["model_class"].dropna().unique())
            palette = dict(zip(classes, CYCLE[: len(classes)]))
            for cls in classes:
                mask = sub["model_class"] == cls
                if mask.any():
                    ax.scatter(
                        x[mask], sub.loc[mask, "universality"],
                        c=palette[cls], alpha=0.6, s=20, label=cls,
                        edgecolors="white", linewidth=0.3,
                    )
            ax.legend(fontsize=7, handletextpad=0.3)
        else:
            ax.scatter(x, sub["universality"], alpha=0.5, s=20)

        z = np.polyfit(x, sub["universality"], 1)
        xline = np.linspace(x.min(), x.max(), 100)
        ax.plot(xline, np.polyval(z, xline), color=GRAY,
                ls="--", alpha=0.7, lw=1)

        ax.set_xlabel(label)
        ax.set_ylabel("Universality")

    return fig


def run_all(df: pd.DataFrame, output_dir: Path) -> dict:
    """Run all regression analyses and generate plots."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    reg = prepare_regression_data(df)
    results = {}

    # 1. Pairwise correlations
    print("  Pairwise correlations:")
    corr_df = pairwise_correlations(reg)
    results["correlations"] = corr_df
    if not corr_df.empty:
        for _, row in corr_df.iterrows():
            print(f"    {row['predictor']:20s}: r={row['pearson_r']:.3f}, p={row['pearson_p']:.4f}, n={row['n']}")
        corr_df.to_csv(out / "pairwise_correlations.csv", index=False)

    # 2. Multiple regression
    print("\n  Multiple regression:")
    reg_result = multiple_regression(reg)
    results["regression"] = reg_result
    if "error" not in reg_result:
        print(f"    R² = {reg_result['r_squared']:.3f}, adj. R² = {reg_result['adj_r_squared']:.3f}, n = {reg_result['n']}")
        print(f"    Predictors: {reg_result['predictors_used']}")
        for _, row in reg_result["coefficients"].iterrows():
            sig = "*" if row["p"] < 0.05 else " "
            print(f"    {sig} {row['predictor']:20s}: beta={row['beta']:.4f}, t={row['t']:.2f}, p={row['p']:.4f}")
        reg_result["coefficients"].to_csv(out / "regression_coefficients.csv", index=False)

    # 3. Partial correlations
    print("\n  Partial correlations:")
    partial_df = partial_correlations(reg)
    results["partial"] = partial_df
    if not partial_df.empty:
        for _, row in partial_df.iterrows():
            sig = "*" if row["p_value"] < 0.05 else " "
            print(f"    {sig} {row['predictor']:20s}: partial r={row['partial_r']:.3f}, r²={row['partial_r_squared']:.3f}, p={row['p_value']:.4f}")
        partial_df.to_csv(out / "partial_correlations.csv", index=False)

    # 4. Variance partitioning
    print("\n  Variance partitioning:")
    vp_df = variance_partitioning(reg)
    results["variance_partitioning"] = vp_df
    if not vp_df.empty:
        for _, row in vp_df.iterrows():
            print(f"    {row['predictor_group']:25s}: alone={row['r2_alone']:.3f}, unique={row['r2_incremental']:.3f}, shared={row['r2_shared']:.3f}")
        vp_df.to_csv(out / "variance_partitioning.csv", index=False)

    # Plots
    fig = plot_correlation_matrix(reg)
    save_figure(fig, out / "correlation_matrix", formats=("pdf",))

    if "error" not in reg_result:
        fig = plot_regression_coefficients(reg_result)
        save_figure(fig, out / "regression_coefficients", formats=("pdf",))

    fig = plot_partial_correlations(partial_df)
    save_figure(fig, out / "partial_correlations", formats=("pdf",))

    fig = plot_variance_partitioning(vp_df)
    save_figure(fig, out / "variance_partitioning", formats=("pdf",))

    fig = plot_scatter_continuous(df)
    save_figure(fig, out / "scatter_continuous", formats=("pdf",))

    return results


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    from experiments.universality.controlled_comparisons.data import load_merged

    setup_plotting()
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)

    df = load_merged(rank=cfg.factorization.rank)
    print(f"Loaded {len(df)} models")
    print("\n=== Regression Analysis ===")
    run_all(df, out)


if __name__ == "__main__":
    main()
