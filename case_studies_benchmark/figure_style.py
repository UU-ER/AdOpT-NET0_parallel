"""
The publication style of the study figures.

One place for the palette, the rcParams and the saving, so that a figure
drawn by any of the plot scripts looks like the others and exports the same
way. The conventions are those of the scientific-figure-making skill:
minimalist axes with the top and the right spine off, a semantic palette
rather than the matplotlib cycle, values annotated on the bars instead of a
dense grid, and a vector file written beside every raster one.

Use it as::

    from figure_style import PALETTE, apply_publication_style, finalize

    apply_publication_style()
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    ...
    finalize(figure, path)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

# Semantic, not decorative. Blue carries the thing being recommended, red the
# thing it is being compared against, green an improvement, neutral the
# background series
PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_light": "#AADCA9",
    "green_strong": "#8BCF8B",
    "red_light": "#E9A6A1",
    "red_strong": "#B64342",
    "neutral": "#CFCECE",
    "grey_text": "#4D4D4D",
    "highlight": "#FFD700",
    "teal": "#42949E",
    "violet": "#9A4D8E",
}

# Compact analytic plots, which is what every figure of this study is. The
# large panel preset of the house style, 24 point on a linewidth of 3, is for
# a slide with four bars on it
FONT_SIZE = 15
AXES_LINEWIDTH = 2.0

DPI = 300


def apply_publication_style(font_size: int = FONT_SIZE):
    """
    Sets the rcParams every figure of the study is drawn with.

    Called once before the figures are created, not per figure.

    :param int font_size: base font size, 15 for a compact analytic panel
    """
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": font_size,
            "axes.linewidth": AXES_LINEWIDTH,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": font_size,
            "axes.labelsize": font_size - 1,
            "xtick.labelsize": font_size - 2,
            "ytick.labelsize": font_size - 2,
            "legend.fontsize": font_size - 2,
            "legend.frameon": False,
            "lines.linewidth": 2.4,
            "lines.markersize": 7,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            # Keeps the text editable in the vector file rather than
            # outlining it, so a label can be fixed without redrawing
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def finalize(figure, output: Path, formats: list = None, pad: float = 2.0):
    """
    Writes a figure to disk in every format asked for.

    A raster file for a slide and a vector file for the paper, from the same
    call, so the two cannot drift apart.

    :param figure: matplotlib Figure
    :param Path output: file path, with or without a suffix
    :param list formats: extensions to write, png and pdf by default
    :param float pad: padding of the final tight_layout pass
    :return: list of the paths written
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(pad=pad)

    written = []
    for suffix in formats or ["png", "pdf"]:
        path = output.with_suffix(f".{suffix}")
        figure.savefig(path, dpi=DPI)
        written.append(path)

    plt.close(figure)
    return written


def annotate_bars(axis, bars, fmt: str = "{:.2f}", fontsize: int = 11):
    """
    Prints the value above each bar.

    The house style puts the numbers on the bars and keeps the grid faint, so
    that a reader takes the exact value off the figure rather than off a grid
    line.

    :param axis: matplotlib Axes the bars live on
    :param bars: BarContainer returned by axis.bar
    :param str fmt: format of the number
    :param int fontsize: size of the annotation
    """
    for bar in bars:
        height = bar.get_height()
        axis.annotate(
            fmt.format(height),
            (bar.get_x() + bar.get_width() / 2, height),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=PALETTE["grey_text"],
        )


def faint_grid(axis, which: str = "y"):
    """
    A grid that supports the numbers without competing with them

    :param axis: matplotlib Axes
    :param str which: axis to draw it on
    """
    axis.grid(axis=which, color="0.88", linewidth=0.8, zorder=0)
    axis.set_axisbelow(True)
