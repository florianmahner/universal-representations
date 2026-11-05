from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.colors import (
    ROSE, TEAL, CYAN, PURPLE,
    GRAY, setup_style,
)

CLASS_COLORS = {
    "Convolutional": ROSE,
    "Transformer": TEAL,
    "MLP-Mixer": CYAN,
    "Hybrid": PURPLE,
}

TASK_MAP = {
    "classification": "Supervised",
    "selfsupervised": "Self-supervised",
    "clip": "CLIP",
    "slip": "SLIP",
    "seer": "SEER",
    "bit_expert": "BiT-expert",
    "taskonomy": "Taskonomy",
    "yolo": "YOLO",
    "detection": "Detection",
    "segmentation": "Segmentation",
    "monoculardepth": "Depth",
    "random": "Random init",
}

DATA_MAP = {
    "imagenet": "ImageNet-1K",
    "imagenet21k": "ImageNet-21K",
    "openai400M": "OpenAI-400M",
    "YFCC15M": "YFCC-15M",
    "random1B": "Random-1B",
    "big_transfer": "BiT transfer",
    "taskonomy": "Taskonomy",
    "coco2017": "COCO",
    "coco,voc": "COCO+VOC",
    "ReDWeb,DIML,Movies,MegaDepth,WSVD,TartanAir,HRWSI,ApolloScape,BlendedMVS,IRS": "Depth mix",
    "dfn2b": "DFN-2B",
    "metaclip_fullcc": "MetaCLIP-FullCC",
    "datacomp_xl": "DataComp-XL",
    "metaclip400m": "MetaCLIP-400M",
    "laion2b": "LAION-2B",
    "laion400m": "LAION-400M",
}

PAPER_WIDTH = 5.5  # NeurIPS text width (139.7mm)
PAPER_FONT = 5            # uniform font size for all paper text
PANEL_LABEL_FONT = 7      # a, b, c panel labels (bold)

SIZES = {
    'single': (3.2, 2.6),
    'square': (2.8, 2.8),
    'wide': (4.5, 2.6),
    'full_width': (PAPER_WIDTH, 2.6),
}

DEFAULT_PAD = {
    'left': 0.6,
    'right': 0.15,
    'bottom': 0.55,
    'top': 0.35,
}


def setup_plotting():
    setup_style()


def paper_style():
    """Apply NeurIPS paper font and figure overrides.

    Sets all text to PAPER_FONT (5pt) and updates SIZES so that
    create_figure() produces figures at NeurIPS-compatible widths:
      single  → 0.5 * textwidth  (include at width=0.5\\linewidth)
      wide    → 0.75 * textwidth (include at width=0.75\\linewidth)
      full    → textwidth        (include at width=\\linewidth)
    """
    setup_plotting()
    plt.rcParams.update({
        "font.size": PAPER_FONT,
        "axes.labelsize": PAPER_FONT,
        "axes.titlesize": PAPER_FONT,
        "axes.titleweight": "medium",
        "xtick.labelsize": PAPER_FONT,
        "ytick.labelsize": PAPER_FONT,
        "legend.fontsize": PAPER_FONT,
        "lines.linewidth": 1.0,
        "lines.markersize": 3,
        "axes.linewidth": 0.4,
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.pad_inches": 0.08,
    })
    SIZES.update({
        "single": (PAPER_WIDTH * 0.5, 2.2),
        "square": (PAPER_WIDTH * 0.5, PAPER_WIDTH * 0.5),
        "wide": (PAPER_WIDTH * 0.75, 2.2),
        "full_width": (PAPER_WIDTH, 2.2),
    })


setup_plotting()


def paper_figure(
    nrows: int = 1,
    ncols: int = 1,
    aspect: float | None = 1.0,
    mosaic=None,
    **kw,
) -> tuple[plt.Figure, plt.Axes | np.ndarray | dict]:
    """Multi-panel paper figure at NeurIPS text width with consistent sizing.

    Height is computed from panel width (PAPER_WIDTH / ncols) × aspect × nrows.
    Panels get set_box_aspect(aspect) so they share identical physical dimensions
    across all figures.

    aspect=1.0  → square panels (default, for scatter/rank plots)
    aspect=0.75 → landscape panels
    aspect=None → no constraint (matplotlib decides)

    Use mosaic= for named axes (passed to subplot_mosaic).
    """
    panel_w = PAPER_WIDTH / ncols
    a = aspect if aspect is not None else 0.7
    fig_h = nrows * (panel_w * a + 0.2)
    kw.setdefault("layout", "constrained")

    if mosaic is not None:
        fig, axes = plt.subplot_mosaic(mosaic, figsize=(PAPER_WIDTH, fig_h), **kw)
        ax_iter = axes.values()
    else:
        fig, axes = plt.subplots(nrows, ncols, figsize=(PAPER_WIDTH, fig_h), **kw)
        ax_iter = np.atleast_1d(axes).flat

    if aspect is not None:
        for ax in ax_iter:
            ax.set_box_aspect(aspect)

    return fig, axes


def create_figure(
    size: str = "single",
    nrows: int = 1,
    ncols: int = 1,
    pad_left: float | None = None,
    pad_right: float | None = None,
    pad_bottom: float | None = None,
    pad_top: float | None = None,
    **subplot_kw,
) -> tuple[plt.Figure, plt.Axes | np.ndarray]:
    """Create figure with fixed plotting-area size."""
    pl = pad_left if pad_left is not None else DEFAULT_PAD['left']
    pr = pad_right if pad_right is not None else DEFAULT_PAD['right']
    pb = pad_bottom if pad_bottom is not None else DEFAULT_PAD['bottom']
    pt = pad_top if pad_top is not None else DEFAULT_PAD['top']

    plot_w, plot_h = SIZES[size]
    total_w = plot_w * ncols
    total_h = plot_h * nrows
    fig_w = pl + total_w + pr
    fig_h = pb + total_h + pt

    fig, ax = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), **subplot_kw)

    fig.subplots_adjust(
        left=pl / fig_w,
        right=(pl + total_w) / fig_w,
        bottom=pb / fig_h,
        top=(pb + total_h) / fig_h,
    )

    return fig, ax


def save_figure(
    fig: plt.Figure,
    path: Path | str,
    close: bool = True,
    tight: bool = True,
    formats: tuple[str, ...] = ("pdf",),
):
    """Save figure in multiple formats."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kw = {"bbox_inches": "tight"} if tight else {}
    for fmt in formats:
        fig.savefig(path.with_suffix(f".{fmt}"), dpi=300, **kw)
    if close:
        plt.close(fig)


def scatter_by_class(
    ax: plt.Axes,
    df,
    x_col: str,
    y_col: str,
    colors: dict[str, str] | None = None,
    fit: bool = True,
    show_legend: bool = True,
    scatter_kw: dict | None = None,
) -> tuple[float, float] | tuple[None, None]:
    """Scatter plot colored by model_class with optional regression line."""
    from scipy.stats import pearsonr

    if colors is None:
        colors = CLASS_COLORS
    kw = dict(alpha=0.6, s=25, edgecolors="white", linewidth=0.3)
    if scatter_kw:
        kw.update(scatter_kw)

    for cls, color in colors.items():
        mask = df["model_class"] == cls
        if mask.any():
            ax.scatter(
                df.loc[mask, x_col], df.loc[mask, y_col],
                c=color, label=f"{cls} (n={mask.sum()})" if show_legend else cls,
                **kw,
            )

    r_val, p_val = None, None
    if fit:
        sub = df.dropna(subset=[x_col, y_col])
        if len(sub) > 2:
            r_val, p_val = pearsonr(sub[x_col], sub[y_col])
            z = np.polyfit(sub[x_col], sub[y_col], 1)
            x_line = np.linspace(sub[x_col].min(), sub[x_col].max(), 100)
            ax.plot(x_line, np.polyval(z, x_line), "--", color=GRAY, lw=1, alpha=0.7)

    if show_legend:
        ax.legend(handletextpad=0.3)

    return r_val, p_val
