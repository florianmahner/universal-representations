from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from experiments.universality.shared import load_components_for_rank, load_universality
from src.data import ThingsDataset

# Ridge penalty grid searched for every target.
ALPHAS = np.logspace(-2, 6, 9)

# The four objective reference measures and the kind of structure each carries.
REFERENCE_SETS = {
    "bert": "semantic",
    "wordnet": "semantic",
    "pixel": "low-level",
    "conv1": "low-level",
}


def sets_of_kind(kind: str) -> tuple[str, ...]:
    """The reference measures of one kind, in their canonical order."""
    return tuple(name for name, k in REFERENCE_SETS.items() if k == kind)


def _pixel_stats_one(path: Path) -> np.ndarray:
    """18 interpretable image statistics: luminance mean/std/entropy, gradient
    magnitude mean/p95 and edge density, per-channel RGB mean/std, saturation
    mean/std, colorfulness, and low/mid/high spatial-frequency power."""
    img = Image.open(path).convert("RGB").resize((128, 128))
    rgb = np.asarray(img, dtype=np.float64) / 255.0
    gray = rgb @ [0.299, 0.587, 0.114]

    hist, _ = np.histogram(gray, bins=64, range=(0, 1), density=False)
    p = hist / hist.sum()
    entropy = -np.sum(p[p > 0] * np.log2(p[p > 0]))

    gy, gx = np.gradient(gray)
    grad = np.hypot(gx, gy)

    sat = rgb.max(axis=2) - rgb.min(axis=2)
    rg = rgb[..., 0] - rgb[..., 1]
    yb = 0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2]
    colorfulness = np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())

    spec = np.abs(np.fft.fftshift(np.fft.fft2(gray - gray.mean()))) ** 2
    freq = np.hypot(*np.meshgrid(
        np.fft.fftshift(np.fft.fftfreq(128, d=1 / 128)),
        np.fft.fftshift(np.fft.fftfreq(128, d=1 / 128)),
    ))
    total = spec.sum() + 1e-12
    bands = [
        spec[(freq >= lo) & (freq < hi)].sum() / total
        for lo, hi in ((0, 8), (8, 32), (32, np.inf))
    ]

    return np.array([
        gray.mean(), gray.std(), entropy, grad.mean(), np.percentile(grad, 95),
        (grad > 0.1).mean(),
        *rgb.mean(axis=(0, 1)), *rgb.std(axis=(0, 1)),
        sat.mean(), sat.std(), colorfulness, *bands,
    ])


def build_pixel_stats(
    info: pd.DataFrame, images_root: Path, features: Path, n_jobs: int
) -> np.ndarray:
    path = features / "pixel_stats_22k.npy"
    if path.exists():
        return np.load(path)
    print(f"Computing pixel statistics for {len(info)} images")
    paths = [
        images_root / cat / fname
        for cat, fname in zip(info["category"], info["filename"])
    ]
    stats = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(_pixel_stats_one)(p) for p in paths
    )
    stats = np.array(stats, dtype=np.float32)
    np.save(path, stats)
    return stats


def _resolve_synsets(concepts: list[str], concepts_tsv: Path) -> list[str | None]:
    """Resolve each concept to a WordNet synset name via its THINGS WordNet ID,
    falling back to a noun lookup of the concept word (offsets from a different
    WordNet version), or None if unresolvable."""
    import warnings

    from nltk.corpus import wordnet as wn

    meta = pd.read_csv(concepts_tsv, sep="\t")
    id_map = dict(zip(meta["uniqueID"], meta["WordNet ID"]))
    word_map = dict(zip(meta["uniqueID"], meta["Word"]))

    names: list[str | None] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for c in concepts:
            synset = None
            m = re.fullmatch(r"n(\d+)", str(id_map.get(c)))
            if m:
                try:
                    synset = wn.synset_from_pos_and_offset("n", int(m.group(1)))
                except Exception:
                    synset = None
            if synset is None:
                word = str(word_map.get(c, c)).replace(" ", "_")
                candidates = wn.synsets(word, pos="n")
                synset = candidates[0] if candidates else None
            names.append(synset.name() if synset is not None else None)
    return names


def _wordnet_row(i: int, names: list[str | None]) -> np.ndarray:
    from nltk.corpus import wordnet as wn

    row = np.full(len(names), np.nan)
    if names[i] is None:
        return row
    a = wn.synset(names[i])
    for j, name in enumerate(names):
        if name is not None:
            row[j] = a.wup_similarity(wn.synset(name)) or 0.0
    return row


def build_wordnet_sim(
    concepts: list[str], concepts_tsv: Path, features: Path, n_jobs: int
) -> np.ndarray:
    path = features / "wordnet_sim_1854.npy"
    if path.exists():
        return np.load(path)
    names = _resolve_synsets(concepts, concepts_tsv)
    n_valid = sum(n is not None for n in names)
    print(f"Computing WordNet Wu-Palmer kernel ({n_valid}/{len(concepts)} synsets)")
    rows = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(_wordnet_row)(i, names) for i in range(len(concepts))
    )
    sim = np.array(rows)
    np.save(path, sim)
    return sim


def kernel_components(sim: np.ndarray, keep_var: float = 0.95) -> np.ndarray:
    """Featurize a similarity kernel via its leading kernel-PCA components.

    A WordNet similarity matrix has no raw feature-vector form, so each concept
    is represented by the top eigenvectors (scaled by sqrt(eigenvalue)) of the
    kernel — kernel-PCA regression, standard for using a similarity matrix as a
    predictor. We keep the leading positive components capturing `keep_var` of
    the positive-eigenvalue mass (the taxonomic structure; the negligible tail
    is dropped for numerical stability). NaN rows (concepts without a synset)
    are imputed with the column mean.
    """
    valid = ~np.isnan(sim).all(axis=1)
    sub = 0.5 * (sim[np.ix_(valid, valid)] + sim[np.ix_(valid, valid)].T)
    vals, vecs = np.linalg.eigh(sub)
    pos = vals[::-1].clip(min=0)
    vecs = vecs[:, ::-1]
    k = int(np.searchsorted(np.cumsum(pos) / pos.sum(), keep_var) + 1)
    print(f"WordNet kernel-PCA: {k} components ({keep_var:.0%} of trace)")
    emb_valid = vecs[:, :k] * np.sqrt(pos[:k])
    emb = np.tile(emb_valid.mean(axis=0), (len(sim), 1))
    emb[valid] = emb_valid
    return emb


def build_predictors(
    info: pd.DataFrame,
    concepts: list[str],
    data_root: Path,
    features: Path,
    n_jobs: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Assemble predictor matrices per set at both evaluation levels.

    Concept level (1,854 rows): semantic references are inherently
    concept-level (word/WordNet similarity); the image-level references (pixel
    stats, untrained conv1) are averaged to concept means, keeping the four
    measures on the same footing and matching the paper's category-level
    embedding. Image level (22,248 rows): pixel/conv1 stay per-image so the
    image-level analysis can exploit within-concept appearance variation
    (bert/wordnet are constant within a concept and are handled by the
    concept-fit-and-broadcast path in compute_scores). bert: 768-d word
    embeddings; wordnet: kernel-PCA components; pixel: 18 interpretable
    statistics; conv1: 128 untrained first-layer feats.
    """
    concept_idx = np.array([concepts.index(c) for c in info["category"]])
    stored = json.loads((features / "concepts.json").read_text())
    assert stored == concepts, "Concept order mismatch with extract_torch.py cache"

    concept_level = {
        "bert": np.load(features / "bert_1854.npy"),
        "wordnet": kernel_components(
            build_wordnet_sim(
                concepts,
                data_root / "meta" / "things22k" / "concepts-metadata_things.tsv",
                features,
                n_jobs,
            )
        ),
    }
    image_level = {
        "pixel": build_pixel_stats(
            info, data_root / "images" / "things", features, n_jobs
        ).astype(np.float64),
        "conv1": np.load(features / "untrained_conv1_22k.npy").astype(np.float64),
    }
    counts = np.bincount(concept_idx, minlength=len(concepts))
    for name, x in image_level.items():
        agg = np.zeros((len(concepts), x.shape[1]), dtype=np.float64)
        np.add.at(agg, concept_idx, x)
        concept_level[name] = agg / counts[:, None]

    return concept_level, image_level


def _cv_r2_multitarget(
    X: np.ndarray, Y: np.ndarray, splits: list, dead: np.ndarray, scale: bool = True
) -> np.ndarray:
    """5-fold ridge R^2 of every column of Y from X, all targets fit at once.

    For each fold: fit RidgeCV (per-target GCV-selected alpha from ALPHAS) on
    all targets at once, then score R^2 on the held-out fold. Mean over folds.
    NaN for dead (constant) targets. `scale` z-scores features on the train
    fold (right for raw feature blocks); it is off for the WordNet kernel-PCA
    features, which already carry the sqrt-eigenvalue scaling of kernel ridge
    and whose low-variance tail components would blow up under standardization.
    """
    r2 = np.zeros((len(splits), Y.shape[1]))
    for f, (tr, va) in enumerate(splits):
        Xtr, Xva = X[tr], X[va]
        if scale:
            scaler = StandardScaler().fit(Xtr)
            Xtr, Xva = scaler.transform(Xtr), scaler.transform(Xva)
        model = RidgeCV(alphas=ALPHAS, alpha_per_target=True).fit(Xtr, Y[tr])
        pred = model.predict(Xva)
        yv = Y[va]
        ss_res = ((yv - pred) ** 2).sum(axis=0)
        ss_tot = ((yv - yv.mean(axis=0)) ** 2).sum(axis=0)
        r2[f] = 1 - ss_res / np.where(ss_tot < 1e-12, np.nan, ss_tot)
    out = r2.mean(axis=0)
    out[dead] = np.nan
    return out


def _img_r2(yv: np.ndarray, pred: np.ndarray) -> np.ndarray:
    ss_res = ((yv - pred) ** 2).sum(axis=0)
    ss_tot = ((yv - yv.mean(axis=0)) ** 2).sum(axis=0)
    return 1 - ss_res / np.where(ss_tot < 1e-12, np.nan, ss_tot)


def _cv_r2_image_broadcast(
    Xc: np.ndarray,
    Yc: np.ndarray,
    W_img: np.ndarray,
    concept_idx: np.ndarray,
    splits: list,
    dead: np.ndarray,
    scale: bool,
) -> np.ndarray:
    """Image-level R^2 for concept-constant features (bert/wordnet).

    With a constant descriptor per concept and 12 images each, an image-level
    ridge fit is exactly a concept-level fit on concept-mean loadings, so the
    model is fit at the concept level — where per-target GCV alpha selection
    is valid, every row being an independent concept — and its predictions are
    broadcast to the held-out concepts' images and scored against the raw
    image loadings. The resulting R^2 is bounded by the dimension's eta2."""
    r2 = np.zeros((len(splits), W_img.shape[1]))
    for f, (tr, va) in enumerate(splits):
        Xtr, Xva = Xc[tr], Xc[va]
        if scale:
            scaler = StandardScaler().fit(Xtr)
            Xtr, Xva = scaler.transform(Xtr), scaler.transform(Xva)
        model = RidgeCV(alphas=ALPHAS, alpha_per_target=True).fit(Xtr, Yc[tr])
        pred_c = model.predict(Xva)
        mask = np.isin(concept_idx, va)
        pred = pred_c[np.searchsorted(va, concept_idx[mask])]
        r2[f] = _img_r2(W_img[mask], pred)
    out = r2.mean(axis=0)
    out[dead] = np.nan
    return out


def _cv_r2_image_direct(
    Xi: np.ndarray,
    W_img: np.ndarray,
    concept_idx: np.ndarray,
    splits: list,
    dead: np.ndarray,
) -> np.ndarray:
    """Image-level R^2 for image-varying features (pixel/conv1).

    Test folds hold out whole concepts. Per-target alpha cannot come from GCV
    (leave-one-image-out keeps the 11 concept-mates in training, biasing alpha
    toward memorizing concept identity), so it is selected on an inner
    concept-grouped split of the train fold, then the ridge is refit on the
    full train fold with the selected per-target alphas via the
    eigendecomposition of X^T X."""

    def _rot(X: np.ndarray, Y: np.ndarray):
        vals, V = np.linalg.eigh(X.T @ X)
        return vals.clip(min=0), V, V.T @ (X.T @ Y)

    r2 = np.zeros((len(splits), W_img.shape[1]))
    for f, (tr_c, va_c) in enumerate(splits):
        tr = np.isin(concept_idx, tr_c)
        va = np.isin(concept_idx, va_c)
        scaler = StandardScaler().fit(Xi[tr])
        Xtr, Xva = scaler.transform(Xi[tr]), scaler.transform(Xi[va])
        Ytr = W_img[tr]

        rng = np.random.default_rng(100 + f)
        inner_va_c = rng.permutation(tr_c)[: max(1, len(tr_c) // 5)]
        iv = np.isin(concept_idx[tr], inner_va_c)
        ym_in = Ytr[~iv].mean(axis=0)
        vals, V, B = _rot(Xtr[~iv], Ytr[~iv] - ym_in)
        Xiv_rot = Xtr[iv] @ V
        sse = np.stack([
            (((Ytr[iv] - ym_in) - Xiv_rot @ (B / (vals + a)[:, None])) ** 2).sum(axis=0)
            for a in ALPHAS
        ])
        alpha_star = ALPHAS[sse.argmin(axis=0)]

        ym = Ytr.mean(axis=0)
        vals2, V2, B2 = _rot(Xtr, Ytr - ym)
        pred = (Xva @ V2) @ (B2 / (vals2[:, None] + alpha_star[None, :])) + ym
        r2[f] = _img_r2(W_img[va], pred)
    out = r2.mean(axis=0)
    out[dead] = np.nan
    return out


def compute_scores(
    cfg: DictConfig,
    concept_level: dict[str, np.ndarray],
    image_level: dict[str, np.ndarray],
    concepts: list[str],
    out: Path,
) -> pd.DataFrame:
    """Score every dimension at the concept level (r2_<set>, concept-mean
    loadings, 5-fold CV over concepts) and at the image level (r2_img_<set>,
    raw 22,248 image loadings, same concept folds so test images always come
    from held-out concepts). Also records eta2, the between-concept fraction
    of each dimension's image-level variance: the ceiling on image-level R^2
    for concept-constant (semantic) features, which within-concept variation
    lets pixel/conv1 exceed."""
    path = out / "grounding_scores.csv"
    cached = pd.read_csv(path) if path.exists() else None
    if cached is not None and all(f"r2_img_{n}" in cached for n in REFERENCE_SETS):
        return cached

    components = load_components_for_rank(
        rank=cfg.factorization.rank,
        models=list(cfg.models) or None,
        dataset=cfg.dataset,
    )
    keys = sorted(components)
    index = [(m, d) for m in keys for d in range(components[m].shape[1])]

    # Component rows are 22,248 image loadings in ThingsDataset order (the
    # image_info_22k.csv row order shared by the cached pixel/conv1 features);
    # average each dimension to the 1,854 basic-level concepts.
    row_concepts = ThingsDataset().all_categories
    concept_pos = {c: i for i, c in enumerate(concepts)}
    concept_idx = np.array([concept_pos[c] for c in row_concepts])
    counts = np.bincount(concept_idx, minlength=len(concepts)).astype(np.float64)

    W_img = np.hstack([components[m] for m in keys]).astype(np.float64)
    Y = np.zeros((len(concepts), W_img.shape[1]))
    np.add.at(Y, concept_idx, W_img)
    Y /= counts[:, None]
    dead = Y.std(axis=0) < 1e-10

    grand = W_img.mean(axis=0)
    ss_tot = ((W_img - grand) ** 2).sum(axis=0)
    ss_between = (counts[:, None] * (Y - grand) ** 2).sum(axis=0)
    eta2 = np.where(ss_tot < 1e-12, np.nan, ss_between / ss_tot)

    df = pd.DataFrame(index, columns=["model", "dimension"])
    df["eta2"] = eta2

    splits = list(KFold(n_splits=5, shuffle=True, random_state=0).split(Y))

    if cached is not None:
        assert (cached["model"] == df["model"]).all() and (
            cached["dimension"] == df["dimension"]
        ).all(), "Cached grounding_scores.csv row order mismatch"
        for name in REFERENCE_SETS:
            df[f"r2_{name}"] = cached[f"r2_{name}"].to_numpy()
    else:
        print(
            f"Concept level: {len(index)} dimensions x {len(REFERENCE_SETS)} sets (RidgeCV)"
        )
        results = Parallel(n_jobs=min(cfg.n_jobs, len(REFERENCE_SETS)), verbose=5)(
            delayed(_cv_r2_multitarget)(concept_level[name], Y, splits, dead, name != "wordnet")
            for name in REFERENCE_SETS
        )
        for name, vals in zip(REFERENCE_SETS, results):
            df[f"r2_{name}"] = vals

    # Image level: same concept folds (test images are always from held-out
    # concepts); semantic sets fit at concept level and broadcast, low-level
    # sets fit per image with inner-grouped alpha selection.
    dead_img = W_img.std(axis=0) < 1e-10
    print(f"Image level: {len(index)} dimensions x {len(REFERENCE_SETS)} sets")
    results_img = Parallel(n_jobs=min(cfg.n_jobs, len(REFERENCE_SETS)), verbose=5)(
        [
            delayed(_cv_r2_image_broadcast)(
                concept_level[name], Y, W_img, concept_idx, splits, dead_img,
                name != "wordnet",
            )
            for name in sets_of_kind("semantic")
        ]
        + [
            delayed(_cv_r2_image_direct)(
                image_level[name], W_img, concept_idx, splits, dead_img
            )
            for name in sets_of_kind("low-level")
        ]
    )
    for name, vals in zip(sets_of_kind("semantic") + sets_of_kind("low-level"), results_img):
        df[f"r2_img_{name}"] = vals

    univ = load_universality(rank=cfg.factorization.rank, dataset=cfg.dataset)[
        ["model", "dimension", "universality"]
    ]
    df = df.merge(univ, on=["model", "dimension"], how="left")
    df.to_csv(path, index=False)
    return df


def load_ratings(data_root: Path) -> pd.DataFrame:
    df = pd.read_csv(data_root / "ratings" / "dimension_ratings.csv")
    parts = df["dim_id"].str.rsplit("__d", n=1, expand=True)
    df["model"] = parts[0]
    df["dimension"] = parts[1].astype(int)
    df["frac_semantic"] = (df["n_semantic"] + df["n_both"]) / df["n_raters"]
    df["frac_visual"] = (df["n_visual"] + df["n_both"]) / df["n_raters"]
    return df


def _boot_auc(labels: np.ndarray, score: np.ndarray, n_boot: int = 2000) -> dict:
    rng = np.random.default_rng(0)
    auc = roc_auc_score(labels, score)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(labels), len(labels))
        if labels[idx].min() == labels[idx].max():
            continue
        boots.append(roc_auc_score(labels[idx], score[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"auc": float(auc), "ci_low": float(lo), "ci_high": float(hi)}


def _spearman(x: pd.Series, y: pd.Series) -> dict:
    valid = x.notna() & y.notna()
    rho, p = spearmanr(x[valid], y[valid])
    return {"rho": float(rho), "p": float(p), "n": int(valid.sum())}


def within_model_rho(df: pd.DataFrame, col: str, min_dims: int = 10) -> dict:
    rhos = []
    for _, grp in df.dropna(subset=["universality", col]).groupby("model"):
        if len(grp) >= min_dims:
            r, _ = spearmanr(grp["universality"], grp[col])
            if np.isfinite(r):
                rhos.append(r)
    rhos = np.array(rhos)
    return {
        "median_rho": float(np.median(rhos)),
        "frac_positive": float((rhos > 0).mean()),
        "wilcoxon_p": float(wilcoxon(rhos).pvalue),
        "n_models": len(rhos),
    }


def analyze(scores: pd.DataFrame, data_root: Path, out: Path) -> dict:
    ratings = load_ratings(data_root)
    rated = ratings.merge(
        scores.drop(columns=["universality"]), on=["model", "dimension"], how="inner"
    )
    rated.to_csv(out / "rated_dims_scores.csv", index=False)
    print(f"Rated dimensions matched to scores: {len(rated)}/{len(ratings)}")

    stats: dict = {"n_dims_scored": int(len(scores)), "n_dims_rated": int(len(rated))}

    sem, low = "r2_sem_mean", "r2_low_mean"
    rated[sem] = rated[[f"r2_{n}" for n in sets_of_kind("semantic")]].mean(axis=1)
    rated[low] = rated[[f"r2_{n}" for n in sets_of_kind("low-level")]].mean(axis=1)

    stats["mean_r2_by_label"] = {
        label: {
            "semantic_r2": float(grp[sem].mean()),
            "lowlevel_r2": float(grp[low].mean()),
            "n": int(len(grp)),
        }
        for label, grp in rated.groupby("majority_label")
    }

    valid = rated.dropna(subset=[sem, low])
    is_sem = valid["majority_label"].isin(["semantic", "both"]).to_numpy()
    is_vis = valid["majority_label"].isin(["visual", "both"]).to_numpy()
    stats["auc_semantic_label"] = _boot_auc(is_sem, valid[sem].to_numpy())
    stats["auc_visual_label"] = _boot_auc(is_vis, valid[low].to_numpy())

    sv = valid[valid["majority_label"].isin(["semantic", "visual"])]
    stats["auc_semantic_vs_visual"] = _boot_auc(
        (sv["majority_label"] == "semantic").to_numpy(),
        (sv[sem] - sv[low]).to_numpy(),
    )
    stats["spearman_frac_semantic"] = _spearman(rated["frac_semantic"], rated[sem])
    stats["spearman_frac_visual"] = _spearman(rated["frac_visual"], rated[low])

    def per_set_stats(prefix: str) -> dict:
        per_set = {}
        for name in REFERENCE_SETS:
            col = f"{prefix}{name}"
            v = rated.dropna(subset=[col])
            per_set[name] = {
                "mean_r2_by_label": {
                    label: float(grp[col].mean())
                    for label, grp in v.groupby("majority_label")
                },
                "auc_semantic_label": _boot_auc(
                    v["majority_label"].isin(["semantic", "both"]).to_numpy(),
                    v[col].to_numpy(),
                ),
                "auc_visual_label": _boot_auc(
                    v["majority_label"].isin(["visual", "both"]).to_numpy(),
                    v[col].to_numpy(),
                ),
                "spearman_frac_semantic": _spearman(rated["frac_semantic"], rated[col]),
                "spearman_frac_visual": _spearman(rated["frac_visual"], rated[col]),
                "universality": {
                    "overall": _spearman(scores["universality"], scores[col]),
                    "within_model": within_model_rho(scores, col),
                },
            }
        return per_set

    stats["per_set"] = per_set_stats("r2_")

    if all(f"r2_img_{n}" in rated for n in REFERENCE_SETS):
        sem_i, low_i = "r2_img_sem_mean", "r2_img_low_mean"
        rated[sem_i] = rated[[f"r2_img_{n}" for n in sets_of_kind("semantic")]].mean(axis=1)
        rated[low_i] = rated[[f"r2_img_{n}" for n in sets_of_kind("low-level")]].mean(axis=1)
        valid_i = rated.dropna(subset=[sem_i, low_i])
        sv_i = valid_i[valid_i["majority_label"].isin(["semantic", "visual"])]
        img: dict = {
            "mean_r2_by_label": {
                label: {
                    "semantic_r2": float(grp[sem_i].mean()),
                    "lowlevel_r2": float(grp[low_i].mean()),
                    "eta2": float(grp["eta2"].mean()),
                    "n": int(len(grp)),
                }
                for label, grp in rated.groupby("majority_label")
            },
            "auc_semantic_label": _boot_auc(
                valid_i["majority_label"].isin(["semantic", "both"]).to_numpy(),
                valid_i[sem_i].to_numpy(),
            ),
            "auc_visual_label": _boot_auc(
                valid_i["majority_label"].isin(["visual", "both"]).to_numpy(),
                valid_i[low_i].to_numpy(),
            ),
            "auc_semantic_vs_visual": _boot_auc(
                (sv_i["majority_label"] == "semantic").to_numpy(),
                (sv_i[sem_i] - sv_i[low_i]).to_numpy(),
            ),
            "spearman_frac_semantic": _spearman(rated["frac_semantic"], rated[sem_i]),
            "spearman_frac_visual": _spearman(rated["frac_visual"], rated[low_i]),
            "per_set": per_set_stats("r2_img_"),
        }
        stats["image_level"] = img

    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def write_report(stats: dict, out: Path) -> None:
    lines = [
        "# Objective grounding of dimension labels",
        "",
        f"Every NMF dimension ({stats['n_dims_scored']} total: 162 models, rank 50) is "
        "scored at the basic-level-category level by how well its concept-mean loadings "
        "are predicted (5-fold cross-validated ridge R2 over the 1,854 THINGS concepts) "
        "by four objective reference measures: two **semantic** — BERT embeddings of the "
        "concept names and a WordNet Wu-Palmer similarity kernel — and two **low-level** — "
        "18 pixel statistics (luminance, contrast, gradients, color, spatial frequency) and "
        "untrained ResNet-50 first-layer features. All four are independent of the THINGS "
        "behavioral data. The 'semantic R2' / 'low-level R2' headline columns are the "
        "per-dimension mean of the two measures of each kind; the individual measures are "
        "broken out below. These scores involve no human judgment of the dimensions.",
        "",
        f"## 1. The human labels track objective properties "
        f"(n={stats['n_dims_rated']} rated dimensions)",
        "",
        "Tests whether the human semantic/visual/both/neither majority labels are "
        "arbitrary or grounded. If the taxonomy is meaningful, dimensions labeled "
        "'semantic' should be predictable from semantic features and dimensions labeled "
        "'visual' from low-level features. The human label is the test target; the "
        "objective scores were computed blind to it.",
        "",
        "| majority label | semantic R2 | low-level R2 | n |",
        "|---|---|---|---|",
    ]
    for label in ("semantic", "both", "visual", "neither"):
        row = stats["mean_r2_by_label"].get(label)
        if row:
            lines.append(
                f"| {label} | {row['semantic_r2']:.3f} | {row['lowlevel_r2']:.3f} | {row['n']} |"
            )
    a1 = stats["auc_semantic_label"]
    a2 = stats["auc_visual_label"]
    a3 = stats["auc_semantic_vs_visual"]
    fs = stats["spearman_frac_semantic"]
    fv = stats["spearman_frac_visual"]
    lines += [
        "",
        "Double dissociation: 'semantic' dimensions score highest on semantic R2 and "
        "lowest on low-level R2, 'visual' dimensions the reverse, 'both' high on both, "
        "'neither' low on both. AUC quantifies label recovery (0.5 = chance; e.g. 0.85 "
        "means that for 85% of random pairs the objective score ranks a semantic-labeled "
        "dimension above a non-semantic-labeled one):",
        "",
        f"- 'semantic'/'both' labels recovered by semantic R2: "
        f"AUC = {a1['auc']:.3f} [{a1['ci_low']:.3f}, {a1['ci_high']:.3f}]",
        f"- 'visual'/'both' labels recovered by low-level R2: "
        f"AUC = {a2['auc']:.3f} [{a2['ci_low']:.3f}, {a2['ci_high']:.3f}]",
        f"- 'semantic' vs 'visual' dimensions separated by (semantic R2 - low-level R2): "
        f"AUC = {a3['auc']:.3f} [{a3['ci_low']:.3f}, {a3['ci_high']:.3f}]",
        f"- The labels are also graded, not binary — the *fraction of raters* choosing "
        f"semantic correlates with semantic R2 (rho = {fs['rho']:.3f}, p = {fs['p']:.2g}) "
        f"and the fraction choosing visual with low-level R2 (rho = {fv['rho']:.3f}, "
        f"p = {fv['p']:.2g})",
        "",
        "### Per individual reference measure",
        "",
        "| measure | R2 sem | R2 both | R2 vis | R2 neither | AUC sem label | AUC vis label |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in REFERENCE_SETS:
        ps = stats["per_set"][name]
        by = ps["mean_r2_by_label"]
        lines.append(
            f"| {name} | {by.get('semantic', float('nan')):.3f} | {by.get('both', float('nan')):.3f} "
            f"| {by.get('visual', float('nan')):.3f} | {by.get('neither', float('nan')):.3f} "
            f"| {ps['auc_semantic_label']['auc']:.3f} | {ps['auc_visual_label']['auc']:.3f} |"
        )
    lines += [
        "",
        "## 2. Universality-semantics replicates without human raters "
        "(all dimensions, labels unused)",
        "",
        "The paper's claim that universal dimensions are semantic originally rested on "
        "the human labels. Here it is re-derived from the objective scores alone: "
        "universality should correlate with semantic R2 but not with low-level R2. "
        "No human rating enters this analysis.",
        "",
        "| measure | overall rho | within-model median rho | frac models > 0 | Wilcoxon p |",
        "|---|---|---|---|---|",
    ]
    for name in REFERENCE_SETS:
        u = stats["per_set"][name]["universality"]
        lines.append(
            f"| {name} | {u['overall']['rho']:.3f} | {u['within_model']['median_rho']:.3f} "
            f"| {u['within_model']['frac_positive']:.2f} | {u['within_model']['wilcoxon_p']:.2g} |"
        )

    img = stats.get("image_level")
    if img:
        lines += [
            "",
            "## 3. Image-level robustness check (exemplar level, held-out concepts)",
            "",
            "The concept-level analysis above averages both the loadings and the "
            "pixel/conv1 features over the 12 exemplars of each concept, which discards "
            "within-concept appearance variation — exactly where visual structure is "
            "dissociated from meaning — and is therefore conservative for the visual "
            "side. Here the targets are instead the raw 22,248 image-level loadings, "
            "using the same 5 concept folds (test images always belong to held-out "
            "concepts). pixel/conv1 stay per-image and can now use within-concept "
            "variation, with per-target alpha chosen on an inner concept-grouped split; "
            "bert/wordnet are constant within a concept (fit at concept level, "
            "predictions broadcast to images), so their image-level R2 is ceilinged by "
            "eta2, the between-concept fraction of a dimension's variance (also "
            "reported: low eta2 = appearance-driven dimension).",
            "",
            "| majority label | semantic R2 (img) | low-level R2 (img) | eta2 | n |",
            "|---|---|---|---|---|",
        ]
        for label in ("semantic", "both", "visual", "neither"):
            row = img["mean_r2_by_label"].get(label)
            if row:
                lines.append(
                    f"| {label} | {row['semantic_r2']:.3f} | {row['lowlevel_r2']:.3f} "
                    f"| {row['eta2']:.3f} | {row['n']} |"
                )
        ia1, ia2, ia3 = (
            img["auc_semantic_label"], img["auc_visual_label"], img["auc_semantic_vs_visual"],
        )
        ifs, ifv = img["spearman_frac_semantic"], img["spearman_frac_visual"]
        lines += [
            "",
            f"- 'semantic'/'both' labels recovered by image-level semantic R2: "
            f"AUC = {ia1['auc']:.3f} [{ia1['ci_low']:.3f}, {ia1['ci_high']:.3f}]",
            f"- 'visual'/'both' labels recovered by image-level low-level R2: "
            f"AUC = {ia2['auc']:.3f} [{ia2['ci_low']:.3f}, {ia2['ci_high']:.3f}]",
            f"- 'semantic' vs 'visual' separated by (semantic - low-level) image R2: "
            f"AUC = {ia3['auc']:.3f} [{ia3['ci_low']:.3f}, {ia3['ci_high']:.3f}]",
            f"- fraction of raters choosing semantic vs image-level semantic R2: "
            f"rho = {ifs['rho']:.3f} (p = {ifs['p']:.2g}); choosing visual vs image-level "
            f"low-level R2: rho = {ifv['rho']:.3f} (p = {ifv['p']:.2g})",
            "",
            "### Per individual reference measure (image level)",
            "",
            "| measure | R2 sem | R2 both | R2 vis | R2 neither | AUC sem label | AUC vis label |",
            "|---|---|---|---|---|---|---|",
        ]
        for name in REFERENCE_SETS:
            ps = img["per_set"][name]
            by = ps["mean_r2_by_label"]
            lines.append(
                f"| {name} | {by.get('semantic', float('nan')):.3f} | {by.get('both', float('nan')):.3f} "
                f"| {by.get('visual', float('nan')):.3f} | {by.get('neither', float('nan')):.3f} "
                f"| {ps['auc_semantic_label']['auc']:.3f} | {ps['auc_visual_label']['auc']:.3f} |"
            )
        lines += [
            "",
            "### Universality (image level)",
            "",
            "| measure | overall rho | within-model median rho | frac models > 0 | Wilcoxon p |",
            "|---|---|---|---|---|",
        ]
        for name in REFERENCE_SETS:
            u = img["per_set"][name]["universality"]
            lines.append(
                f"| {name} | {u['overall']['rho']:.3f} | {u['within_model']['median_rho']:.3f} "
                f"| {u['within_model']['frac_positive']:.2f} | {u['within_model']['wilcoxon_p']:.2g} |"
            )

    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)
    # Cached reference-measure predictors, shared with extract_torch.py.
    features = out / "features"
    features.mkdir(parents=True, exist_ok=True)

    info = pd.read_csv(
        Path(cfg.data_root) / "meta" / "things22k" / "image_info_22k.csv"
    )
    concepts = sorted(info["category"].unique())

    concept_level, image_level = build_predictors(
        info, concepts, Path(cfg.data_root), features, cfg.n_jobs
    )
    scores = compute_scores(cfg, concept_level, image_level, concepts, out)
    write_report(analyze(scores, Path(cfg.data_root), out), out)


if __name__ == "__main__":
    main()
