#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from omegaconf import OmegaConf
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from src.colors import GRAY, GRAY_DARK, GRAY_PALE, INDIGO, TEAL
from src.plotting import PANEL_LABEL_FONT, PAPER_WIDTH, paper_style

N_TOPICS = 16
HUMAN_LABEL_ORDER = ["semantic", "visual", "both", "neither"]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Malformed JSON in {path} at line {line_number}: {error}"
                ) from error
            record["_record_index"] = line_number
            records.append(record)
    return records


def load_descriptions(
    path: Path,
    *,
    prompt_version: str,
    image_mode: str,
    expected_dimensions: int = 1_059,
    expected_records: int = 3_150,
    min_generator_support: int = 2,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the latest successful description for each dimension and model."""
    if not path.exists():
        raise FileNotFoundError(f"Description log not found: {path}")

    rows: list[dict[str, Any]] = []
    for record in _read_jsonl(path):
        response = record.get("response")
        if (
            record.get("stage") != "description"
            or record.get("prompt_version") != prompt_version
            or record.get("image_mode") != image_mode
            or not isinstance(response, dict)
        ):
            continue
        text = response.get("overall_description")
        if not isinstance(text, str) or not text.strip():
            continue
        rows.append(
            {
                "dim_id": record.get("dim_id"),
                "description_generator": record.get("requested_model"),
                "resolved_model": record.get("resolved_model"),
                "prompt_version": record.get("prompt_version"),
                "image_mode": record.get("image_mode"),
                "created_at_utc": record.get("created_at_utc", ""),
                "_record_index": record["_record_index"],
                "overall_description": " ".join(text.split()),
            }
        )

    descriptions = pd.DataFrame(rows)
    if descriptions.empty:
        raise ValueError("No valid stage-1 description records were found")
    if descriptions[["dim_id", "description_generator"]].isna().any().any():
        raise ValueError("Description records contain missing identities")

    descriptions = (
        descriptions.sort_values(["created_at_utc", "_record_index"])
        .drop_duplicates(
            [
                "dim_id",
                "description_generator",
                "prompt_version",
                "image_mode",
            ],
            keep="last",
        )
        .sort_values(["dim_id", "description_generator"])
        .reset_index(drop=True)
    )

    n_dimensions = int(descriptions["dim_id"].nunique())
    if n_dimensions != expected_dimensions:
        raise ValueError(
            f"Expected {expected_dimensions:,} described dimensions, "
            f"found {n_dimensions:,}"
        )
    if len(descriptions) != expected_records:
        raise ValueError(
            f"Expected {expected_records:,} latest description "
            f"records, found {len(descriptions):,}"
        )
    counts = descriptions.groupby("dim_id")["description_generator"].nunique()
    if (counts < min_generator_support).any():
        raise ValueError("Every dimension must have at least two descriptions")
    if descriptions.duplicated(["dim_id", "description_generator"]).any():
        raise ValueError("Latest description table contains duplicate keys")

    diagnostics = {
        "n_records": int(len(descriptions)),
        "n_dimensions": n_dimensions,
        "n_generators": int(descriptions["description_generator"].nunique()),
        "records_per_generator": {
            str(key): int(value)
            for key, value in descriptions["description_generator"]
            .value_counts()
            .sort_index()
            .items()
        },
        "dimensions_by_generator_count": {
            str(int(key)): int(value)
            for key, value in counts.value_counts().sort_index().items()
        },
        "prompt_version": prompt_version,
        "image_mode": image_mode,
    }
    return descriptions, diagnostics


def load_dimension_summary(
    path: Path, *, expected_dimensions: int = 1_059
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dimension summary not found: {path}")
    frame = pd.read_csv(path)
    required = {
        "dim_id",
        "model",
        "universality",
        "majority_label",
        "n_raters",
        "n_semantic",
        "n_visual",
        "n_both",
        "n_neither",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Dimension summary is missing: {', '.join(missing)}")
    if len(frame) != expected_dimensions or frame["dim_id"].nunique() != len(frame):
        raise ValueError(
            f"Dimension summary must contain {expected_dimensions:,} "
            "unique dimensions"
        )
    frame = frame.copy()
    frame["universality"] = pd.to_numeric(frame["universality"], errors="coerce")
    if frame["universality"].isna().any():
        raise ValueError("Universality contains missing or non-numeric values")
    frame["human_label"] = (
        frame["majority_label"].str.lower().replace({"mixed": "both"})
    )
    unknown = sorted(set(frame["human_label"]) - set(HUMAN_LABEL_ORDER))
    if unknown:
        raise ValueError(f"Unexpected human labels: {unknown}")
    vote_columns = [
        "n_raters",
        "n_semantic",
        "n_visual",
        "n_both",
        "n_neither",
    ]
    votes = frame[vote_columns].apply(pd.to_numeric, errors="coerce")
    if votes.isna().any().any():
        raise ValueError("Human vote counts contain missing values")
    if not (
        votes[["n_semantic", "n_visual", "n_both", "n_neither"]].sum(axis=1)
        == votes["n_raters"]
    ).all():
        raise ValueError("Human vote counts do not sum to n_raters")
    return frame


def _make_vectorizer(*, min_df: int) -> TfidfVectorizer:
    # The extra stop words remove boilerplate about the stimulus/prompt while
    # retaining content words such as color, texture, objects, people, food,
    # and backgrounds.
    custom_stop_words = """
        8x8 across appears collection common consists depicted depicting depicts
        description different displaying diverse each feature featured features
        featuring grid highly image images include includes including individual
        many mostly often overall pattern patterns photograph photographs photo
        photographed photographic photos predominantly primarily related show
        showing shown shows subject subjects thematic theme unifying variety
        various visual visually wide
        """.split()
    return TfidfVectorizer(
        stop_words=sorted(set(ENGLISH_STOP_WORDS) | set(custom_stop_words)),
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=0.65,
        max_features=5_000,
        sublinear_tf=True,
        norm="l2",
    )


def _term_anchor_score(term: str, anchor: str) -> float:
    if term == anchor:
        return 1.0
    term_tokens = set(term.split())
    anchor_tokens = set(anchor.split())
    if anchor_tokens.issubset(term_tokens):
        return 0.75
    if term_tokens & anchor_tokens:
        return 0.30
    return 0.0


def match_theme_names(
    components: np.ndarray, terms: np.ndarray
) -> tuple[dict[int, str], dict[int, float]]:
    """One-to-one match NMF components to transparent anchor definitions."""
    # Curated display names are matched to topics through their anchor
    # vocabulary, so NMF component ordering is not assumed.  The exact top
    # phrases and matching score are exported beside every display name.
    theme_definitions: tuple[tuple[str, tuple[str, ...]], ...] = (
        (
            "Everyday objects on plain backgrounds",
            ("everyday", "isolated", "product", "simple backgrounds", "catalog"),
        ),
        (
            "No single shared subject",
            ("unrelated", "animals", "scenes", "objects animals", "no single"),
        ),
        (
            "Prepared foods and dishes",
            ("dishes", "prepared", "baked", "ingredients", "food"),
        ),
        (
            "Vehicles and sports equipment",
            ("vehicles", "sports", "equipment", "gear", "outdoor"),
        ),
        (
            "Body parts and personal-care items",
            ("body parts", "human body", "hands", "care", "skin"),
        ),
        (
            "Furniture and appliances",
            ("furniture", "appliances", "indoor", "household", "storage"),
        ),
        (
            "Fruit and vegetables",
            ("fruits", "vegetables", "produce", "fresh", "nuts"),
        ),
        (
            "Instruments, clocks, and dials",
            ("instruments", "musical", "circular", "dials", "round"),
        ),
        (
            "Small colorful items in groups",
            ("small", "colorful", "collections", "arranged", "candies"),
        ),
        (
            "Clothing and accessories",
            ("clothing", "accessories", "fashion", "personal", "wearable"),
        ),
        (
            "Blue skies and tall structures",
            ("sky", "blue", "structures", "poles", "vertical"),
        ),
        (
            "Warm brown and earthy colors",
            ("warm", "brown", "earthy", "tones", "tan"),
        ),
        (
            "Bottles, jars, and glassware",
            ("containers", "bottles", "glass", "jars", "liquids"),
        ),
        (
            "Tools and metal hardware",
            ("hardware", "metal", "implements", "tools", "handheld"),
        ),
        (
            "Outdoor plants and animals",
            ("natural", "plants", "animals", "green", "nature"),
        ),
        (
            "Materials and surface textures",
            ("textures", "materials", "textured", "texture", "surfaces"),
        ),
    )
    score = np.zeros((components.shape[0], len(theme_definitions)), dtype=float)
    for topic_index, weights in enumerate(components):
        for definition_index, (_, anchors) in enumerate(theme_definitions):
            anchor_weights = 0.0
            for term_index, term in enumerate(terms):
                match = max(
                    _term_anchor_score(str(term), anchor) for anchor in anchors
                )
                if match:
                    anchor_weights += float(weights[term_index]) * match
            score[topic_index, definition_index] = anchor_weights

    row_indices, column_indices = linear_sum_assignment(-score)
    name_by_topic = {
        int(topic): theme_definitions[int(definition)][0]
        for topic, definition in zip(row_indices, column_indices, strict=True)
    }
    score_by_topic = {
        int(topic): float(score[topic, definition])
        for topic, definition in zip(row_indices, column_indices, strict=True)
    }
    if len(name_by_topic) != components.shape[0]:
        raise RuntimeError("Could not assign one readable name per NMF topic")
    return name_by_topic, score_by_topic


def _nmf_stability(
    matrix: Any,
    *,
    n_topics: int,
    seed: int,
    n_runs: int = 6,
) -> dict[str, Any]:
    """Matched component cosine similarity across random initializations."""
    components: list[np.ndarray] = []
    errors: list[float] = []
    for run_index in range(n_runs):
        model = NMF(
            n_components=n_topics,
            init="random",
            random_state=seed + run_index,
            max_iter=1_500,
        )
        model.fit(matrix)
        components.append(normalize(model.components_))
        errors.append(float(model.reconstruction_err_))
    reference = components[0]
    matched_similarities: list[float] = []
    for candidate in components[1:]:
        similarity = reference @ candidate.T
        rows, columns = linear_sum_assignment(-similarity)
        matched_similarities.append(float(similarity[rows, columns].mean()))
    return {
        "n_random_initializations": n_runs,
        "mean_matched_component_cosine": float(
            np.mean(matched_similarities)
        ),
        "sd_matched_component_cosine": float(
            np.std(matched_similarities, ddof=1)
        ),
        "mean_reconstruction_error": float(np.mean(errors)),
    }


def fit_description_themes(
    descriptions: pd.DataFrame,
    *,
    seed: int,
    top_k_per_description: int = 2,
    min_generator_support: int = 2,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    """Fit dimension-level themes and require cross-generator text support."""
    aggregate = (
        descriptions.groupby("dim_id", as_index=False)["overall_description"]
        .agg(" ".join)
        .rename(columns={"overall_description": "combined_description"})
        .sort_values("dim_id")
        .reset_index(drop=True)
    )
    vectorizer = _make_vectorizer(min_df=6)
    dimension_matrix = vectorizer.fit_transform(
        aggregate["combined_description"]
    )
    terms = np.asarray(vectorizer.get_feature_names_out())
    if dimension_matrix.shape[1] < N_TOPICS:
        raise ValueError("Too few text features for the requested NMF model")

    topic_model = NMF(
        n_components=N_TOPICS,
        init="nndsvda",
        random_state=seed,
        max_iter=1_500,
    )
    dimension_weights = topic_model.fit_transform(dimension_matrix)
    individual_weights = topic_model.transform(
        vectorizer.transform(descriptions["overall_description"])
    )
    name_by_topic, name_score_by_topic = match_theme_names(
        topic_model.components_, terms
    )

    top_topic_indices = np.argsort(-individual_weights, axis=1)[
        :, :top_k_per_description
    ]
    description_topics = descriptions[
        ["dim_id", "description_generator"]
    ].copy()
    description_topics["top_topics"] = [
        tuple(int(value) for value in row) for row in top_topic_indices
    ]
    description_topics["top_1_topic"] = top_topic_indices[:, 0]

    support_rows: list[dict[str, Any]] = []
    for dim_id, group in description_topics.groupby("dim_id", sort=True):
        counts = np.zeros(N_TOPICS, dtype=int)
        for topics in group["top_topics"]:
            for topic in topics:
                counts[int(topic)] += 1
        for topic_index, count in enumerate(counts):
            support_rows.append(
                {
                    "dim_id": dim_id,
                    "topic_index": topic_index,
                    "generator_support_count": int(count),
                    "consensus_theme": bool(count >= min_generator_support),
                }
            )
    support = pd.DataFrame(support_rows)

    weight_rows: list[dict[str, Any]] = []
    for dimension_index, dim_id in enumerate(aggregate["dim_id"]):
        order = np.argsort(-dimension_weights[dimension_index])
        rank_by_topic = np.empty(N_TOPICS, dtype=int)
        rank_by_topic[order] = np.arange(1, N_TOPICS + 1)
        for topic_index in range(N_TOPICS):
            weight_rows.append(
                {
                    "dim_id": dim_id,
                    "topic_index": topic_index,
                    "dimension_topic_weight": float(
                        dimension_weights[dimension_index, topic_index]
                    ),
                    "dimension_topic_rank": int(rank_by_topic[topic_index]),
                }
            )
    assignments = support.merge(
        pd.DataFrame(weight_rows),
        on=["dim_id", "topic_index"],
        validate="one_to_one",
    )
    assignments["theme_name"] = assignments["topic_index"].map(name_by_topic)

    top_terms_by_topic: dict[int, list[str]] = {}
    for topic_index, weights in enumerate(topic_model.components_):
        top_terms_by_topic[topic_index] = [
            str(value) for value in terms[np.argsort(weights)[-10:][::-1]]
        ]

    primary_topic = np.argmax(dimension_weights, axis=1)
    aggregate["primary_topic_index"] = primary_topic
    aggregate["primary_theme"] = [
        name_by_topic[int(topic)] for topic in primary_topic
    ]
    consensus_wide = (
        assignments[assignments["consensus_theme"]]
        .sort_values(
            ["dim_id", "dimension_topic_weight"],
            ascending=[True, False],
        )
        .groupby("dim_id")["theme_name"]
        .agg(lambda values: "; ".join(values))
        .rename("consensus_themes")
        .reset_index()
    )
    aggregate = aggregate.merge(
        consensus_wide, on="dim_id", how="left", validate="one_to_one"
    )
    aggregate["consensus_themes"] = aggregate["consensus_themes"].fillna("")

    top_one_agreement = (
        description_topics.groupby("dim_id")["top_1_topic"]
        .agg(lambda values: values.value_counts().max() >= 2)
        .mean()
    )
    at_least_one_consensus = (
        assignments.groupby("dim_id")["consensus_theme"].any().mean()
    )
    stability = _nmf_stability(
        dimension_matrix,
        n_topics=N_TOPICS,
        seed=seed,
    )
    diagnostics = {
        "n_topics": N_TOPICS,
        "top_topics_per_description": top_k_per_description,
        "minimum_generator_support": min_generator_support,
        "vocabulary_size": int(len(terms)),
        "document_term_shape": [
            int(dimension_matrix.shape[0]),
            int(dimension_matrix.shape[1]),
        ],
        "final_reconstruction_error": float(topic_model.reconstruction_err_),
        "fraction_dimensions_with_top_1_generator_agreement": float(
            top_one_agreement
        ),
        "fraction_dimensions_with_at_least_one_consensus_theme": float(
            at_least_one_consensus
        ),
        "stability": stability,
        "theme_name_matching": {
            name_by_topic[index]: {
                "topic_index": int(index),
                "anchor_match_score": float(name_score_by_topic[index]),
                "top_terms": top_terms_by_topic[index],
            }
            for index in sorted(name_by_topic)
        },
    }

    topic_metadata = pd.DataFrame(
        [
            {
                "topic_index": topic_index,
                "theme_name": name_by_topic[topic_index],
                "anchor_match_score": name_score_by_topic[topic_index],
                "top_terms": " | ".join(top_terms_by_topic[topic_index]),
            }
            for topic_index in range(N_TOPICS)
        ]
    )
    return aggregate, assignments, topic_metadata, diagnostics


def compute_lexical_consensus(
    descriptions: pd.DataFrame,
) -> pd.DataFrame:
    """Mean pairwise TF--IDF cosine among generators for each dimension."""
    vectorizer = _make_vectorizer(min_df=3)
    matrix = vectorizer.fit_transform(descriptions["overall_description"])
    rows: list[dict[str, Any]] = []
    for dim_id, indices in descriptions.groupby("dim_id").groups.items():
        index_array = np.asarray(list(indices), dtype=int)
        similarity = cosine_similarity(matrix[index_array])
        pairwise = similarity[np.triu_indices(len(index_array), k=1)]
        rows.append(
            {
                "dim_id": dim_id,
                "n_description_generators": int(len(index_array)),
                "lexical_consensus": float(pairwise.mean()),
                "lexical_consensus_min_pair": float(pairwise.min()),
                "lexical_consensus_max_pair": float(pairwise.max()),
            }
        )
    consensus = pd.DataFrame(rows).sort_values("dim_id").reset_index(drop=True)
    if len(consensus) != descriptions["dim_id"].nunique():
        raise RuntimeError("Lexical consensus does not cover all dimensions")
    return consensus


def _spearman(x: Iterable[float], y: Iterable[float]) -> float:
    x_array = np.asarray(list(x), dtype=float)
    y_array = np.asarray(list(y), dtype=float)
    valid = np.isfinite(x_array) & np.isfinite(y_array)
    x_array = x_array[valid]
    y_array = y_array[valid]
    if (
        len(x_array) < 3
        or np.unique(x_array).size < 2
        or np.unique(y_array).size < 2
    ):
        return math.nan
    return float(
        np.corrcoef(rankdata(x_array), rankdata(y_array))[0, 1]
    )


def bootstrap_lexical_relationship(
    frame: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
    n_bins: int = 10,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Cluster-bootstrap lexical consensus--universality association."""
    data = frame[
        ["dim_id", "model", "universality", "lexical_consensus"]
    ].dropna().reset_index(drop=True)
    clusters = data["model"].drop_duplicates().to_numpy()
    cluster_indices = {
        cluster: data.index[data["model"] == cluster].to_numpy()
        for cluster in clusters
    }
    data["universality_bin"] = pd.qcut(
        data["universality"], q=n_bins, labels=False, duplicates="drop"
    ).astype(int)
    bin_ids = sorted(data["universality_bin"].unique().tolist())

    rng = np.random.default_rng(seed)
    rho_samples = np.empty(n_bootstrap, dtype=float)
    bin_samples = {
        bin_id: np.full(n_bootstrap, np.nan, dtype=float) for bin_id in bin_ids
    }
    for bootstrap_index in range(n_bootstrap):
        sampled_clusters = rng.choice(
            clusters, size=len(clusters), replace=True
        )
        sampled_indices = np.concatenate(
            [cluster_indices[cluster] for cluster in sampled_clusters]
        )
        sampled = data.loc[sampled_indices]
        rho_samples[bootstrap_index] = _spearman(
            sampled["universality"], sampled["lexical_consensus"]
        )
        means = sampled.groupby("universality_bin")[
            "lexical_consensus"
        ].mean()
        for bin_id in bin_ids:
            if bin_id in means:
                bin_samples[bin_id][bootstrap_index] = float(means[bin_id])

    if not np.isfinite(rho_samples).all():
        raise RuntimeError("Non-finite lexical-consensus bootstrap estimate")
    ci_low, ci_high = np.quantile(rho_samples, [0.025, 0.975])
    observed = _spearman(data["universality"], data["lexical_consensus"])
    scipy_result = spearmanr(
        data["universality"], data["lexical_consensus"]
    )

    binned_rows: list[dict[str, Any]] = []
    for bin_id, group in data.groupby("universality_bin", sort=True):
        samples = bin_samples[int(bin_id)]
        samples = samples[np.isfinite(samples)]
        low, high = np.quantile(samples, [0.025, 0.975])
        binned_rows.append(
            {
                "bin": int(bin_id) + 1,
                "n_dimensions": int(len(group)),
                "universality_mean": float(group["universality"].mean()),
                "universality_min": float(group["universality"].min()),
                "universality_max": float(group["universality"].max()),
                "lexical_consensus_mean": float(
                    group["lexical_consensus"].mean()
                ),
                "lexical_consensus_ci_low": float(low),
                "lexical_consensus_ci_high": float(high),
            }
        )
    result = {
        "n_dimensions": int(len(data)),
        "n_source_models": int(len(clusters)),
        "spearman_rho": float(observed),
        "spearman_p_value": float(scipy_result.pvalue),
        "cluster_bootstrap_ci_low": float(ci_low),
        "cluster_bootstrap_ci_high": float(ci_high),
        "n_bootstrap": int(n_bootstrap),
        "bootstrap_unit": "source_model",
        "bootstrap_seed": int(seed),
    }
    return result, pd.DataFrame(binned_rows)


def make_theme_summary(
    assignments: pd.DataFrame,
    topic_metadata: pd.DataFrame,
    dimension_data: pd.DataFrame,
) -> pd.DataFrame:
    joined = assignments.merge(
        dimension_data[
            ["dim_id", "universality", "human_label", "combined_description"]
        ],
        on="dim_id",
        how="left",
        validate="many_to_one",
    )
    n_dimensions = int(dimension_data["dim_id"].nunique())
    rows: list[dict[str, Any]] = []
    for topic_index, group in joined.groupby("topic_index", sort=True):
        consensus = group[group["consensus_theme"]].copy()
        metadata = topic_metadata[
            topic_metadata["topic_index"] == topic_index
        ].iloc[0]
        representative = consensus.sort_values(
            "dimension_topic_weight", ascending=False
        ).head(3)
        row: dict[str, Any] = {
            "topic_index": int(topic_index),
            "theme_name": metadata["theme_name"],
            "top_terms": metadata["top_terms"],
            "anchor_match_score": float(metadata["anchor_match_score"]),
            "n_consensus_dimensions": int(len(consensus)),
            "consensus_prevalence": float(len(consensus) / n_dimensions),
            "mean_universality": float(consensus["universality"].mean()),
            "median_universality": float(consensus["universality"].median()),
        }
        for label in HUMAN_LABEL_ORDER:
            count = int((consensus["human_label"] == label).sum())
            row[f"n_human_{label}"] = count
            row[f"fraction_human_{label}"] = (
                float(count / len(consensus)) if len(consensus) else math.nan
            )
        for example_index, (_, example) in enumerate(
            representative.iterrows(), start=1
        ):
            row[f"representative_dim_{example_index}"] = example["dim_id"]
            row[f"representative_description_{example_index}"] = example[
                "combined_description"
            ]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["consensus_prevalence", "theme_name"],
        ascending=[False, True],
    )


def make_description_atlas(
    descriptions: pd.DataFrame,
    dimension_data: pd.DataFrame,
) -> pd.DataFrame:
    model_column = {
        "google/gemini-3.6-flash": "description_gemini_3_6_flash",
        "anthropic/claude-sonnet-5": "description_claude_sonnet_5",
        "openai/gpt-5.6-luna": "description_gpt_5_6_luna",
    }
    wide = descriptions.pivot(
        index="dim_id",
        columns="description_generator",
        values="overall_description",
    )
    wide = wide.rename(columns=model_column).reset_index()
    for output_column in model_column.values():
        if output_column not in wide:
            wide[output_column] = ""
    columns = [
        "dim_id",
        "model",
        "universality",
        "human_label",
        "n_raters",
        "n_semantic",
        "n_visual",
        "n_both",
        "n_neither",
        "n_description_generators",
        "lexical_consensus",
        "lexical_consensus_min_pair",
        "lexical_consensus_max_pair",
        "primary_theme",
        "consensus_themes",
    ]
    atlas = dimension_data[columns].merge(
        wide, on="dim_id", how="left", validate="one_to_one"
    )
    return atlas.sort_values("dim_id").reset_index(drop=True)


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.20,
        1.06,
        label,
        transform=axis.transAxes,
        fontsize=PANEL_LABEL_FONT,
        fontweight="bold",
        va="top",
    )


def _quiet_axis(axis: plt.Axes, *, grid_axis: str | None = None) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(False)
    if grid_axis is not None:
        axis.grid(
            axis=grid_axis, color=GRAY_PALE, linewidth=0.4, zorder=0
        )
        axis.set_axisbelow(True)


def plot_description_content(
    theme_summary: pd.DataFrame,
    dimension_data: pd.DataFrame,
    binned: pd.DataFrame,
    lexical_result: dict[str, Any],
    analysis_dir: Path,
) -> list[str]:
    paper_style()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(PAPER_WIDTH, 4.70),
        gridspec_kw={"width_ratios": [1.12, 1.0]},
    )
    fig.subplots_adjust(
        left=0.235,
        right=0.985,
        bottom=0.245,
        top=0.87,
        wspace=0.55,
    )

    axis = axes[0]
    prevalence = theme_summary.sort_values(
        "consensus_prevalence", ascending=True
    )
    y = np.arange(len(prevalence))
    bars = axis.barh(
        y,
        prevalence["consensus_prevalence"],
        height=0.66,
        color=[
            GRAY if name == "No single shared subject" else INDIGO
            for name in prevalence["theme_name"]
        ],
        alpha=0.80,
        edgecolor="white",
        linewidth=0.3,
    )
    axis.set(
        yticks=y,
        yticklabels=prevalence["theme_name"],
        xlim=(0, max(0.25, prevalence["consensus_prevalence"].max() + 0.035)),
        xlabel="Dimensions described with each pattern",
        title="What the descriptions say\nthe dimensions capture",
    )
    axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    for bar, value in zip(
        bars, prevalence["consensus_prevalence"], strict=True
    ):
        axis.text(
            value + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.0%}",
            va="center",
            ha="left",
            color=GRAY_DARK,
            fontsize=4.4,
        )
    _quiet_axis(axis, grid_axis="x")
    _panel_label(axis, "a")

    axis = axes[1]
    axis.scatter(
        dimension_data["universality"],
        dimension_data["lexical_consensus"],
        s=4.0,
        color=TEAL,
        alpha=0.12,
        edgecolors="none",
        rasterized=True,
        zorder=1,
    )
    x = binned["universality_mean"].to_numpy()
    y_values = binned["lexical_consensus_mean"].to_numpy()
    y_error = np.vstack(
        [
            y_values - binned["lexical_consensus_ci_low"].to_numpy(),
            binned["lexical_consensus_ci_high"].to_numpy() - y_values,
        ]
    )
    axis.errorbar(
        x,
        y_values,
        yerr=y_error,
        color=TEAL,
        marker="o",
        markerfacecolor="white",
        markeredgecolor=TEAL,
        markeredgewidth=0.7,
        markersize=3.2,
        linewidth=1.1,
        capsize=1.5,
        zorder=3,
    )
    axis.text(
        0.04,
        0.96,
        (
            f"Strength of upward relationship: "
            f"{lexical_result['spearman_rho']:.2f}\n"
            rf"95% interval: "
            rf"{lexical_result['cluster_bootstrap_ci_low']:.2f}–"
            rf"{lexical_result['cluster_bootstrap_ci_high']:.2f}"
        ),
        transform=axis.transAxes,
        ha="left",
        va="top",
        color=GRAY_DARK,
    )
    axis.set(
        xlabel=(
            "Universality score\n"
            "(higher = shared more consistently across vision models)"
        ),
        ylabel=(
            "Overlap in words and phrases across descriptions\n"
            "(higher = more similarly worded)"
        ),
        title="More universal dimensions receive\n"
        "more similarly worded descriptions",
    )
    axis.set_ylim(
        -0.01,
        max(0.50, float(dimension_data["lexical_consensus"].max()) + 0.02),
    )
    _quiet_axis(axis, grid_axis="both")
    _panel_label(axis, "b")

    fig.text(
        0.03,
        0.018,
        (
            "Left: A dimension enters a bar when that pattern is among the "
            "two strongest in descriptions from at least two LLMs; one "
            "dimension can enter several bars.\n"
            "Right: Faint dots are individual dimensions; connected circles "
            "average ten equally sized groups. Similarity is overlap in "
            "informative words and short phrases.\n"
            "These are patterns in what the LLMs wrote, not independently "
            "verified image properties. Thin vertical lines around the "
            "circles show 95% uncertainty intervals."
        ),
        ha="left",
        va="bottom",
        color=GRAY_DARK,
        fontsize=4.2,
    )

    analysis_dir.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    for suffix in ("pdf", "png"):
        path = analysis_dir / f"description_content.{suffix}"
        fig.savefig(
            path,
            dpi=300,
            facecolor="white",
            bbox_inches=None,
            pad_inches=0,
        )
        files.append(str(path))
    plt.close(fig)
    return files


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def run_analysis(
    output_dir: Path,
    *,
    seed: int,
    n_bootstrap: int = 5_000,
    prompt_version: str = "neutral_patterns_v1",
    image_mode: str = "grid",
) -> dict[str, Any]:
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap must be at least 100")
    output_dir = output_dir.resolve()
    analysis_dir = output_dir / "additional_analyses"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    descriptions, description_diagnostics = load_descriptions(
        output_dir / "descriptions.jsonl",
        prompt_version=prompt_version,
        image_mode=image_mode,
    )
    summary = load_dimension_summary(output_dir / "dimension_summary.csv")
    aggregate, assignments, topic_metadata, theme_diagnostics = (
        fit_description_themes(descriptions, seed=seed)
    )
    lexical = compute_lexical_consensus(descriptions)

    dimension_data = (
        summary.merge(
            aggregate, on="dim_id", how="inner", validate="one_to_one"
        )
        .merge(lexical, on="dim_id", how="inner", validate="one_to_one")
        .sort_values("dim_id")
        .reset_index(drop=True)
    )
    if len(dimension_data) != len(summary):
        raise RuntimeError("Description-content joins dropped dimensions")

    lexical_result, binned = bootstrap_lexical_relationship(
        dimension_data,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    lexical_result["complete_three_generator_sensitivity"] = {
        "n_dimensions": int(
            (dimension_data["n_description_generators"] == 3).sum()
        ),
        "spearman_rho": _spearman(
            dimension_data.loc[
                dimension_data["n_description_generators"] == 3,
                "universality",
            ],
            dimension_data.loc[
                dimension_data["n_description_generators"] == 3,
                "lexical_consensus",
            ],
        ),
    }
    lexical_result["within_human_label"] = {}
    for label in HUMAN_LABEL_ORDER:
        subset = dimension_data[dimension_data["human_label"] == label]
        lexical_result["within_human_label"][label] = {
            "n_dimensions": int(len(subset)),
            "spearman_rho": _spearman(
                subset["universality"], subset["lexical_consensus"]
            ),
        }

    theme_summary = make_theme_summary(
        assignments,
        topic_metadata,
        dimension_data,
    )
    atlas = make_description_atlas(descriptions, dimension_data)
    figure_paths = plot_description_content(
        theme_summary,
        dimension_data,
        binned,
        lexical_result,
        analysis_dir,
    )

    theme_summary_path = analysis_dir / "content_theme_summary.csv"
    assignments_path = analysis_dir / "content_theme_assignments.csv"
    atlas_path = analysis_dir / "description_text_atlas.csv"
    binned_path = analysis_dir / "lexical_consensus_binned.csv"
    summary_path = analysis_dir / "description_content_summary.json"
    theme_summary.to_csv(theme_summary_path, index=False)
    assignments.sort_values(["dim_id", "topic_index"]).to_csv(
        assignments_path, index=False
    )
    atlas.to_csv(atlas_path, index=False)
    binned.to_csv(binned_path, index=False)

    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "literal_description_content",
        "sources": {
            "descriptions_jsonl": str(
                (output_dir / "descriptions.jsonl").resolve()
            ),
            "dimension_summary_csv": str(
                (output_dir / "dimension_summary.csv").resolve()
            ),
        },
        "measurement": {
            "theme_model": (
                "TF-IDF word unigrams/bigrams followed by 16-component NMF "
                "on one concatenated document per dimension"
            ),
            "theme_support_rule": (
                "A theme counts only when it is among the top two NMF topics "
                "for at least two independent description generators"
            ),
            "lexical_consensus": (
                "Mean pairwise TF-IDF cosine similarity among the available "
                "overall descriptions for a dimension"
            ),
            "uncertainty": (
                f"95% percentile intervals from {n_bootstrap:,} source-model "
                "cluster bootstrap samples"
            ),
            "interpretation_boundary": (
                "Themes and lexical convergence characterize the literal "
                "description corpus. They are not independent confirmation "
                "that a named property is visible in the image grid."
            ),
        },
        "description_diagnostics": description_diagnostics,
        "theme_diagnostics": theme_diagnostics,
        "lexical_consensus_result": lexical_result,
        "theme_prevalence": theme_summary[
            [
                "theme_name",
                "n_consensus_dimensions",
                "consensus_prevalence",
                "mean_universality",
            ]
        ].to_dict(orient="records"),
        "outputs": {
            "theme_summary_csv": str(theme_summary_path),
            "theme_assignments_csv": str(assignments_path),
            "description_atlas_csv": str(atlas_path),
            "lexical_binned_csv": str(binned_path),
            "figures": figure_paths,
            "summary_json": str(summary_path),
        },
    }
    summary_path.write_text(
        json.dumps(_json_ready(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)
    manifest = run_analysis(out, seed=cfg.seed)
    result = manifest["lexical_consensus_result"]
    print(
        f"Validated {manifest['description_diagnostics']['n_dimensions']:,} "
        "dimensions and "
        f"{manifest['description_diagnostics']['n_records']:,} descriptions."
    )
    print(
        "Lexical consensus vs universality: "
        f"rho={result['spearman_rho']:.4f}, model-bootstrap 95% CI "
        f"[{result['cluster_bootstrap_ci_low']:.4f}, "
        f"{result['cluster_bootstrap_ci_high']:.4f}]"
    )
    print(
        "Wrote description themes, atlas, statistics, and figures under "
        f"{Path(manifest['outputs']['summary_json']).parent}"
    )


if __name__ == "__main__":
    main()
