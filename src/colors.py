import os
import colorsys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Named colors

ROSE   = "#CC6677"
TEAL   = "#44AA99"
CYAN   = "#88CCEE"
SAND   = "#DDCC77"
PURPLE = "#AA4499"

# Full Tol Muted (all 10) for when you need more
INDIGO = "#332288"
GREEN  = "#117733"
WINE   = "#882255"
OLIVE  = "#999933"

# Grays
GRAY        = "#999999"   # medium gray for reference lines, secondary data
GRAY_LIGHT  = "#CCCCCC"   # light gray for bars, backgrounds
GRAY_DARK   = "#555555"   # dark gray for text-like elements
GRAY_PALE   = "#E8E8E8"   # very light for shaded regions

# Pre-built combos

CYCLE  = [ROSE, TEAL, INDIGO, SAND, PURPLE]        # default color cycle (lightness-spread)
PAIR   = [ROSE, TEAL]                              # primary 2-color pair
TRIPLE = [ROSE, TEAL, INDIGO]                      # + dark anchor
QUAD   = [ROSE, TEAL, INDIGO, SAND]                # + light anchor
FIVE   = [ROSE, TEAL, INDIGO, SAND, PURPLE]        # full set

# Alternative pairs
PAIR_ALT1 = [PURPLE, CYAN]
PAIR_ALT2 = [PURPLE, SAND]
PAIR_ALT3 = [ROSE, CYAN]

# Color manipulation helpers


def lighten(hex_color, amount=0.5):
    """Mix a color toward white. amount=0 returns original, amount=1 returns white."""
    r, g, b = mcolors.to_rgb(hex_color)
    return mcolors.to_hex((r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount))


def desaturate(hex_color, factor=0.7):
    """Reduce saturation. factor=1 is original, factor=0 is grayscale."""
    r, g, b = mcolors.to_rgb(hex_color)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    r2, g2, b2 = colorsys.hsv_to_rgb(h, s * factor, v)
    return mcolors.to_hex((r2, g2, b2))


def soft(hex_color):
    """Create a soft fill version: lightened 35% + desaturated 20%.

    Use for bar fills, area charts, violin fills — anywhere you want
    the color to be present but not overpowering. Pair with the original
    color as edgecolor for a polished look:

        plt.bar(x, y, color=soft(ROSE), edgecolor=ROSE, linewidth=1.5)
    """
    return lighten(desaturate(hex_color, 0.8), 0.35)


# Pre-computed soft versions for convenience
SOFT_ROSE   = soft(ROSE)
SOFT_TEAL   = soft(TEAL)
SOFT_INDIGO = soft(INDIGO)
SOFT_SAND   = soft(SAND)
SOFT_PURPLE = soft(PURPLE)
SOFT_CYCLE  = [soft(c) for c in CYCLE]


# Colormaps

# Diverging: Tol BuRd — for RSMs, difference matrices, anything centered at 0
CMAP_DIV = mcolors.LinearSegmentedColormap.from_list("tol_burd", [
    "#2166AC", "#4393C3", "#92C5DE", "#D1E5F0", "#F7F7F7",
    "#FDDBC7", "#F4A582", "#D6604D", "#B2182B",
], N=256)

# Sequential: Tol Sunset — for correlation matrices, similarity values
CMAP_SEQ = mcolors.LinearSegmentedColormap.from_list("tol_sunset", [
    "#364B9A", "#4A7BB7", "#6EA6CD", "#98CAE1", "#C2E4EF",
    "#EAECCC", "#FEDA8B", "#FDB366", "#F67E4B", "#DD3D2D", "#A50026",
], N=256)

# Sequential: Tol Iridescent — subtle, for less dramatic heatmaps
CMAP_IRID = mcolors.LinearSegmentedColormap.from_list("tol_iridescent", [
    "#FEFBE9", "#F5F3C1", "#DDECBF", "#C2E3D2", "#A8D8DC",
    "#8DCBE4", "#7BBCE7", "#88A5DD", "#9B8AC4", "#9A709E",
    "#805770", "#46353A",
], N=256)

# Gray sequential — for when color would be distracting
CMAP_GRAY = mcolors.LinearSegmentedColormap.from_list("seq_gray", [
    "#F5F5F5", "#D9D9D9", "#BDBDBD", "#969696", "#737373",
    "#525252", "#252525",
], N=256)

# Register all custom cmaps so they work by name string too
for _name, _cmap in [("tol_burd", CMAP_DIV), ("tol_sunset", CMAP_SEQ),
                      ("tol_iridescent", CMAP_IRID), ("seq_gray", CMAP_GRAY)]:
    try:
        plt.colormaps.register(_cmap, name=_name)
    except (ValueError, AttributeError):
        pass  # already registered or older matplotlib


# Style setup

def setup_style():
    """Apply the paper matplotlib style and color cycle."""
    # Try to load the .mplstyle file if it exists nearby
    style_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "style.mplstyle")
    if os.path.exists(style_path):
        plt.style.use(style_path)
    else:
        # Fallback: set the essentials directly
        plt.rcParams.update({
            "axes.prop_cycle": plt.cycler("color", CYCLE),
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
            "lines.markeredgewidth": 0.6,
            "lines.markeredgecolor": "white",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "figure.facecolor": "white",
            "image.cmap": "tol_burd",
        })
