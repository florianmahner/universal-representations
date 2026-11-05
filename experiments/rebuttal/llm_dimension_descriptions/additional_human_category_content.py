#!/usr/bin/env python3
from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import PercentFormatter
from omegaconf import OmegaConf

from src.colors import GRAY_DARK, GRAY_PALE, INDIGO, ROSE, TEAL
from src.plotting import PAPER_WIDTH, paper_style

# The four response options of the human rating study, in the order used
# everywhere below, with their display name, color, one-line gloss and the
# full explanation printed on the atlas pages.
LABELS: dict[str, dict[str, str]] = {
    "semantic": {
        "display": "Semantic",
        "color": INDIGO,
        "title_suffix": "same kind of thing",
        "explanation": (
            "Same kind of thing, despite different appearances. Humans judged "
            "that meaning best explained why the 64 images belong together."
        ),
    },
    "visual": {
        "display": "Visual",
        "color": ROSE,
        "title_suffix": "same visible look",
        "explanation": (
            "Same visible look, despite showing different kinds of things. Humans "
            "judged that appearance best explained why the images belong together."
        ),
    },
    "both": {
        "display": "Both",
        "color": TEAL,
        "title_suffix": "same kind of thing and look",
        "explanation": (
            "Same kind of thing and the same visible look. Humans judged that both "
            "meaning and appearance explained the dimension."
        ),
    },
    "neither": {
        "display": "Neither",
        "color": GRAY_DARK,
        "title_suffix": "no single shared property",
        "explanation": (
            "No single shared property covers the whole grid. Humans found neither "
            "one meaning nor one visible feature that explained the dimension."
        ),
    },
}

# Curated from high-human-agreement dimensions after inspecting the actual
# grids.  The population summary, not these examples, supports frequency
# claims.  Captions state only properties visible in the grid and repeated in
# the independent LLM descriptions.
CURATED_EXAMPLES: dict[str, list[dict[str, str]]] = {
    "semantic": [
        {
            "dim_id": "slip_vit_b_clip_cc12m__d7",
            "caption": (
                "Household and outdoor tools: different-looking objects linked "
                "mainly by what they are used for."
            ),
        },
        {
            "dim_id": "timm_convnext_large_fb_in1k__d23",
            "caption": (
                "Prepared foods: visually varied dishes grouped by their "
                "shared food meaning."
            ),
        },
        {
            "dim_id": "slip_vit_b_slip__d45",
            "caption": (
                "Clothing and accessories: shirts, bags, hats, shoes, and "
                "other wearable items."
            ),
        },
    ],
    "visual": [
        {
            "dim_id": "timm_convnext_base_fb_in22k__d13",
            "caption": (
                "Different objects sharing a circular or ring-like shape, "
                "often viewed from above."
            ),
        },
        {
            "dim_id": "taskonomy_egomotion__d23",
            "caption": (
                "Unrelated objects sharing vivid red, pink, and magenta colors."
            ),
        },
        {
            "dim_id": "taskonomy_point_matching__d2",
            "caption": (
                "Unrelated subjects shown with similar warm brown and tan "
                "colors, often in close-up."
            ),
        },
    ],
    "both": [
        {
            "dim_id": "clip_rn101__d13",
            "caption": (
                "Plants and flowers share a meaning and also a predominantly "
                "green appearance."
            ),
        },
        {
            "dim_id": "bit_expert_mammal__d40",
            "caption": (
                "Foods and ingredients share a meaning and also strong red, "
                "pink, or purple coloring."
            ),
        },
        {
            "dim_id": "bit_expert_arthropod__d41",
            "caption": (
                "Nuts, seeds, and dried foods share a meaning and also similar "
                "brown, granular textures."
            ),
        },
    ],
    "neither": [
        {
            "dim_id": "taskonomy_normal__d34",
            "caption": (
                "Animals, foods, people, tools, and outdoor scenes with "
                "no single stable subject or appearance."
            ),
        },
        {
            "dim_id": "taskonomy_jigsaw__d36",
            "caption": (
                "A broad mix of animals, vehicles, foods, household objects, "
                "and scenes; recurring cues cover only small subsets."
            ),
        },
        {
            "dim_id": "midas_dpt_hybrid__d2",
            "caption": (
                "A miscellaneous mix of objects and scenes with no feature "
                "that clearly spans the whole grid."
            ),
        },
    ],
}


def load_data(output_dir: Path, grid_dir: Path) -> pd.DataFrame:
    summary_path = output_dir / "dimension_summary.csv"
    atlas_path = output_dir / "additional_analyses" / "description_text_atlas.csv"
    if not summary_path.exists() or not atlas_path.exists():
        raise FileNotFoundError(
            "Run the main interpretation pipeline and description-content "
            f"analysis first. Missing {summary_path} or {atlas_path}."
        )

    summary = pd.read_csv(summary_path)
    atlas = pd.read_csv(atlas_path)
    frame = summary.merge(
        atlas[
            [
                "dim_id",
                "primary_theme",
                "consensus_themes",
                "lexical_consensus",
                "description_claude_sonnet_5",
                "description_gemini_3_6_flash",
                "description_gpt_5_6_luna",
            ]
        ],
        on="dim_id",
        how="inner",
        validate="one_to_one",
    )
    # The rated dimension set has 1,044 entries; a different count means the
    # two tables were produced from different runs.
    if len(frame) != 1_059:
        raise ValueError(f"Expected 1,044 joined dimensions, found {len(frame):,}.")
    unknown = sorted(set(frame["majority_label"]) - set(LABELS))
    if unknown:
        raise ValueError(f"Unexpected human labels: {unknown}")

    curated_ids = {
        example["dim_id"]
        for examples in CURATED_EXAMPLES.values()
        for example in examples
    }
    missing_rows = sorted(curated_ids - set(frame["dim_id"]))
    missing_grids = sorted(
        dim_id for dim_id in curated_ids if not (grid_dir / f"{dim_id}.png").exists()
    )
    if missing_rows or missing_grids:
        raise FileNotFoundError(
            f"Missing curated rows={missing_rows}; grids={missing_grids}"
        )
    return frame


def compute_pattern_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label in LABELS:
        within = frame[frame["majority_label"] == label]
        outside = frame[frame["majority_label"] != label]
        within_counts = within["primary_theme"].value_counts()
        outside_counts = outside["primary_theme"].value_counts()
        themes = sorted(set(within_counts.index) | set(outside_counts.index))
        for theme in themes:
            n_within = int(within_counts.get(theme, 0))
            n_outside = int(outside_counts.get(theme, 0))
            within_fraction = n_within / len(within)
            outside_fraction = n_outside / len(outside)
            rows.append(
                {
                    "human_label": label,
                    "n_dimensions": len(within),
                    "description_pattern": theme,
                    "n_within_category": n_within,
                    "within_category_fraction": within_fraction,
                    "n_other_categories": n_outside,
                    "other_categories_fraction": outside_fraction,
                    "difference_vs_other_categories": (
                        within_fraction - outside_fraction
                    ),
                }
            )

    result = pd.DataFrame(rows)
    result["rank_within_category"] = (
        result.groupby("human_label")["difference_vs_other_categories"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    result["shown_in_figure"] = result["rank_within_category"] <= 5
    return result.sort_values(
        ["human_label", "rank_within_category"],
        key=lambda series: (
            series.map({label: i for i, label in enumerate(LABELS)})
            if series.name == "human_label"
            else series
        ),
    )


def _quiet_bar_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="x", color=GRAY_PALE, linewidth=0.45)
    axis.set_axisbelow(True)
    axis.tick_params(axis="y", length=0)


def plot_pattern_summary(
    pattern_summary: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    paper_style()
    fig, axes = plt.subplots(4, 1, figsize=(PAPER_WIDTH, 7.15))
    fig.subplots_adjust(
        left=0.40,
        right=0.975,
        bottom=0.095,
        top=0.91,
        hspace=0.72,
    )

    for axis, label in zip(axes, LABELS, strict=True):
        subset = pattern_summary[
            (pattern_summary["human_label"] == label)
            & pattern_summary["shown_in_figure"]
        ].sort_values("within_category_fraction", ascending=True)
        y = np.arange(len(subset))
        bars = axis.barh(
            y,
            subset["within_category_fraction"],
            height=0.62,
            color=LABELS[label]["color"],
            alpha=0.82,
            edgecolor="white",
            linewidth=0.3,
        )
        labels = [
            textwrap.fill(str(value), width=34)
            for value in subset["description_pattern"]
        ]
        axis.set(
            yticks=y,
            yticklabels=labels,
            xlim=(0, 0.25),
            xticks=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25],
            title=(
                f"Humans chose {LABELS[label]['display']}: "
                f"{LABELS[label]['title_suffix']} "
                f"(n={int(subset['n_dimensions'].iloc[0]):,})"
            ),
        )
        axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        for bar, value in zip(
            bars, subset["within_category_fraction"], strict=True
        ):
            axis.text(
                value + 0.004,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.0%}",
                ha="left",
                va="center",
                fontsize=5.2,
                color=GRAY_DARK,
            )
        _quiet_bar_axis(axis)

    axes[-1].set_xlabel("Dimensions in this human category")
    fig.suptitle(
        "What appears in dimensions given each original human label",
        x=0.04,
        y=0.982,
        ha="left",
        fontsize=7.2,
        fontweight="medium",
    )
    fig.text(
        0.04,
        0.945,
        (
            "Each dimension received three independent open descriptions. "
            "The bars show recurring subjects or visible properties that were "
            "especially common within each human category."
        ),
        ha="left",
        va="top",
        fontsize=5.2,
        color=GRAY_DARK,
    )
    fig.text(
        0.04,
        0.018,
        (
            "Shown are the five description patterns that occur more often in "
            "this category than in the other three. Percentages use all "
            "dimensions in the named category; patterns summarize generated "
            "descriptions rather than independent ground-truth labels."
        ),
        ha="left",
        va="bottom",
        fontsize=4.45,
        color=GRAY_DARK,
        wrap=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for suffix in ("pdf", "png"):
        path = output_dir / f"content_by_human_category.{suffix}"
        fig.savefig(path, dpi=300, facecolor="white", pad_inches=0)
        paths.append(str(path))
    plt.close(fig)
    return paths


def _human_vote_text(row: pd.Series, label: str) -> str:
    return (
        f"{int(row[f'n_{label}'])} of {int(row['n_raters'])} human raters "
        f"chose {LABELS[label]['display']}"
    )


def build_example_table(
    frame: pd.DataFrame,
    grid_dir: Path,
) -> pd.DataFrame:
    indexed = frame.set_index("dim_id")
    rows: list[dict[str, Any]] = []
    for label in LABELS:
        for example_number, example in enumerate(
            CURATED_EXAMPLES[label], start=1
        ):
            dim_id = example["dim_id"]
            source = indexed.loc[dim_id]
            if source["majority_label"] != label:
                raise ValueError(
                    f"{dim_id} has human label {source['majority_label']}, "
                    f"expected {label}."
                )
            rows.append(
                {
                    "human_label": label,
                    "example_number": example_number,
                    "dim_id": dim_id,
                    "grid_path": str((grid_dir / f"{dim_id}.png").resolve()),
                    "plain_language_caption": example["caption"],
                    "human_vote_summary": _human_vote_text(source, label),
                    "n_raters": int(source["n_raters"]),
                    "n_semantic": int(source["n_semantic"]),
                    "n_visual": int(source["n_visual"]),
                    "n_both": int(source["n_both"]),
                    "n_neither": int(source["n_neither"]),
                    "universality": float(source["universality"]),
                    "primary_description_pattern": source["primary_theme"],
                    "description_claude_sonnet_5": source[
                        "description_claude_sonnet_5"
                    ],
                    "description_gemini_3_6_flash": source[
                        "description_gemini_3_6_flash"
                    ],
                    "description_gpt_5_6_luna": source[
                        "description_gpt_5_6_luna"
                    ],
                }
            )
    return pd.DataFrame(rows)


def _plot_atlas_page(
    label: str,
    examples: pd.DataFrame,
) -> plt.Figure:
    paper_style()
    fig = plt.figure(figsize=(PAPER_WIDTH, 5.30))
    grid = fig.add_gridspec(
        3,
        2,
        left=0.04,
        right=0.98,
        bottom=0.035,
        top=0.83,
        width_ratios=[1.45, 3.35],
        hspace=0.15,
        wspace=0.08,
    )

    fig.suptitle(
        (
            f"{LABELS[label]['display']} dimensions "
            f"(n={int(examples['category_size'].iloc[0]):,})"
        ),
        x=0.04,
        y=0.978,
        ha="left",
        fontsize=8.0,
        fontweight="medium",
    )
    fig.text(
        0.04,
        0.925,
        textwrap.fill(LABELS[label]["explanation"], width=100),
        ha="left",
        va="top",
        fontsize=5.6,
        color=GRAY_DARK,
    )

    for row_index, (_, row) in enumerate(examples.iterrows()):
        image_axis = fig.add_subplot(grid[row_index, 0])
        image_axis.imshow(mpimg.imread(row["grid_path"]))
        image_axis.set_xticks([])
        image_axis.set_yticks([])
        for spine in image_axis.spines.values():
            spine.set_color(GRAY_PALE)
            spine.set_linewidth(0.5)

        text_axis = fig.add_subplot(grid[row_index, 1])
        text_axis.axis("off")
        text_axis.text(
            0.0,
            0.91,
            f"Example {int(row['example_number'])}",
            ha="left",
            va="top",
            fontsize=6.1,
            fontweight="bold",
            color=LABELS[label]["color"],
        )
        text_axis.text(
            0.0,
            0.69,
            textwrap.fill(row["plain_language_caption"], width=69),
            ha="left",
            va="top",
            fontsize=6.0,
            color="black",
            linespacing=1.25,
        )
        text_axis.text(
            0.0,
            0.31,
            row["human_vote_summary"],
            ha="left",
            va="top",
            fontsize=5.0,
            color=GRAY_DARK,
        )
        text_axis.text(
            0.0,
            0.13,
            f"Dimension: {row['dim_id']}",
            ha="left",
            va="top",
            fontsize=4.1,
            color=GRAY_DARK,
        )

    return fig


def plot_example_atlas(
    frame: pd.DataFrame,
    grid_dir: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, list[str]]:
    example_table = build_example_table(frame, grid_dir)
    category_sizes = frame["majority_label"].value_counts()
    example_table["category_size"] = example_table["human_label"].map(
        category_sizes
    )

    paths: list[str] = []
    atlas_path = output_dir / "human_category_examples_atlas.pdf"
    with PdfPages(atlas_path) as atlas_pdf:
        for label in LABELS:
            examples = example_table[
                example_table["human_label"] == label
            ].sort_values("example_number")
            fig = _plot_atlas_page(label, examples)
            atlas_pdf.savefig(fig, dpi=300, facecolor="white", pad_inches=0)
            for suffix in ("pdf", "png"):
                path = output_dir / f"examples_{label}.{suffix}"
                fig.savefig(path, dpi=300, facecolor="white", pad_inches=0)
                paths.append(str(path))
            plt.close(fig)
    paths.insert(0, str(atlas_path))
    return example_table, paths


def run_analysis(output_dir: Path, grid_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    grid_dir = grid_dir.resolve()
    analysis_dir = output_dir / "additional_analyses"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    frame = load_data(output_dir, grid_dir)
    pattern_summary = compute_pattern_summary(frame)
    pattern_csv = analysis_dir / "content_by_human_category.csv"
    pattern_summary.to_csv(pattern_csv, index=False)
    pattern_figures = plot_pattern_summary(pattern_summary, analysis_dir)

    example_table, atlas_figures = plot_example_atlas(
        frame, grid_dir, analysis_dir
    )
    example_csv = analysis_dir / "human_category_examples.csv"
    example_table.to_csv(example_csv, index=False)

    manifest = {
        "n_dimensions": int(len(frame)),
        "human_category_counts": {
            label: int((frame["majority_label"] == label).sum())
            for label in LABELS
        },
        "selection_note": (
            "The summary uses all dimensions. Atlas examples are illustrative, "
            "curated from high-human-agreement dimensions after inspecting the "
            "actual grids, and cover different recurring content patterns."
        ),
        "outputs": {
            "pattern_csv": str(pattern_csv),
            "example_csv": str(example_csv),
            "pattern_figures": pattern_figures,
            "atlas_figures": atlas_figures,
        },
    }
    manifest_path = analysis_dir / "human_category_content_manifest.json"
    manifest["outputs"]["manifest"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    del cfg  # the analysis is fully specified by the tables run.py wrote
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)
    manifest = run_analysis(
        out, Path("experiments/universality/notebooks/grid_cache_dimensions")
    )
    print(
        f"Validated {manifest['n_dimensions']:,} dimensions; wrote a "
        "four-category content summary and example atlas."
    )
    print(
        "Outputs: "
        f"{Path(manifest['outputs']['manifest']).parent.resolve()}"
    )


if __name__ == "__main__":
    main()
