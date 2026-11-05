from __future__ import annotations

import logging
from itertools import combinations

import numpy as np
import pandas as pd
from nltk.corpus import wordnet as wn

log = logging.getLogger(__name__)


def _get_synset(category: str) -> wn.synset | None:
    """Map a THINGS category name to its first noun synset."""
    # strip disambiguation suffixes like "bat1", "bow2"
    name = category.rstrip("0123456789")
    for query in [name, name.replace("_", "")]:
        synsets = wn.synsets(query, pos=wn.NOUN)
        if synsets:
            return synsets[0]
    return None


def _build_synset_map(categories: np.ndarray) -> dict[str, wn.synset | None]:
    """Build category -> synset mapping for all unique categories."""
    unique = sorted(set(categories))
    mapping = {c: _get_synset(c) for c in unique}
    n_found = sum(v is not None for v in mapping.values())
    log.info("WordNet mapping: %d/%d categories matched (%.0f%%)",
             n_found, len(unique), 100 * n_found / len(unique))
    return mapping


def semantic_coherence(cats: list[str], synset_map: dict) -> float:
    """Mean pairwise Wu-Palmer similarity for a list of categories."""
    synsets = [synset_map[c] for c in cats if synset_map.get(c) is not None]
    if len(synsets) < 2:
        return np.nan
    sims = [s1.wup_similarity(s2) for s1, s2 in combinations(synsets, 2)
            if s1.wup_similarity(s2) is not None]
    return float(np.mean(sims)) if sims else np.nan


def mean_wordnet_depth(cats: list[str], synset_map: dict) -> float:
    """Mean min-depth in WordNet hierarchy for a list of categories."""
    depths = [min(synset_map[c].min_depth(), 20)
              for c in cats if synset_map.get(c) is not None]
    return float(np.mean(depths)) if depths else np.nan


def analyze_semantic_content(
    profiles: pd.DataFrame,
    prototypes: dict[str, np.ndarray],
    categories: np.ndarray,
    n_top: int = 20,
) -> pd.DataFrame:
    """Add semantic coherence and depth columns to cluster profiles.

    For each cluster, takes the top-n highest-loading unique categories
    from the prototype and computes WordNet-based metrics.
    """
    synset_map = _build_synset_map(categories)

    records = []
    for _, row in profiles.iterrows():
        cid = str(row["cluster"])
        proto = prototypes.get(cid)
        if proto is None:
            continue

        # get top-n unique categories by weight
        sorted_idx = np.argsort(-proto)
        seen: set[str] = set()
        top_cats: list[str] = []
        for i in sorted_idx:
            cat = categories[i]
            if cat not in seen:
                seen.add(cat)
                top_cats.append(cat)
                if len(top_cats) >= n_top:
                    break

        records.append({
            "cluster": cid,
            "wn_coherence": semantic_coherence(top_cats, synset_map),
            "wn_depth": mean_wordnet_depth(top_cats, synset_map),
            "top_cats_for_wn": ", ".join(top_cats[:10]),
        })

    result = pd.DataFrame(records)
    log.info("Semantic analysis: %d clusters, coherence range [%.3f, %.3f]",
             len(result), result["wn_coherence"].min(), result["wn_coherence"].max())
    return result
