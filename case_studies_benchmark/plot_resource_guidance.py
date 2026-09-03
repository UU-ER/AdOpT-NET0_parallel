"""
Draws what the benchmark still supports, once the run-to-run noise is counted.

The typical-day clustering is unseeded, so two runs of one configuration come
out a few thousandths of a percent apart in the matrix and up to a factor 5.9
apart in runtime. That noise floor swallows most timing results, but not
everything, and these three figures are what is left:

- cores requested against cores actually used, which is what a
  ``--cpus-per-task`` request should be based on
- peak memory against model size, which is what a ``--mem`` request should be
  based on
- what each modelling knob costs, in time and in memory, with the noise floor
  drawn behind it, so that a box sitting inside the shading reads as unmeasured
  rather than as small

Examples::

    python plot_resource_guidance.py
    python plot_resource_guidance.py --dataset benchmark_dataset.csv
"""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
DATASET_FILE = BASE / "benchmark_dataset.csv"
FIGURES_PATH = BASE / "figures"

BLUE, ORANGE, RED = "#3b6ea5", "#d1731f", "#c0392b"
GREEN, DARK, GREY = "#4a8c5f", "#2b2b2b", "#9a9a9a"

# Run-to-run spread measured on the three configurations that were run twice at
# four threads, as a factor: the geometric mean of x2.21, x4.57 and x5.89 for
# runtime, and of x1.08, x1.03 and x1.15 for memory
NOISE = {"gurobi_runtime_s": 3.9, "rss_peak_os_mb": 1.09}

# Each knob, the token that marks it in a case name, and what its absence means.
# A knob that is at its default never appears in the name
KNOBS = {
    "storage": ("st0", "off", "on"),
    "bidirectional_precise": ("bp1", "on", "off"),
    "electricity_price": ("pf", "fluctuating", "constant"),
    "hydrogen_demand": ("df", "fluctuating", "constant"),
    "pipeline_capex": ("cxfix", "fixed_plus_linear", "linear"),
    "pipeline_size_min": ("sm250", "250", "0"),
    "pv": ("pv0", "off", "on"),
}

# Which setting of each knob is the expensive one, which fixes the direction of
# the ratio: a number above 1 always means the knob cost something
EXPENSIVE = {
    "storage": "on",
    "bidirectional_precise": "on",
    "electricity_price": "fluctuating",
    "hydrogen_demand": "fluctuating",
    "pipeline_capex": "fixed_plus_linear",
    "pipeline_size_min": "250",
    "pv": "on",
}

LABELS = {
    "storage": "storage",
    "pipeline_size_min": "pipeline size_min",
    "hydrogen_demand": "demand fluctuating",
    "bidirectional_precise": "bidirectional precise",
    "electricity_price": "price fluctuating",
    "pipeline_capex": "capex fixed+linear",
    "pv": "pv",
}


def _parse(case_name: str):
    """
    Reads the knob settings back out of a case name

    :param str case_name: name of the run, as built by run_benchmark
    :return: dict of knob name to setting, plus typicaldays and threads
    """
    body = case_name.replace("four_node_", "")
    threads = re.search(r"thr(\d+)", body)
    affinity = re.search(r"aff(\d+)", body)

    settings = {
        "td": int(re.search(r"td(\d+)", body).group(1)),
        "thr": int(threads.group(1)) if threads else 0,
        "aff": int(affinity.group(1)) if affinity else 0,
    }

    tokens = set(body.split("_"))
    for knob, (token, present, absent) in KNOBS.items():
        settings[knob] = present if token in tokens else absent

    return settings


def load(dataset_file: Path):
    """
    Reads the dataset and adds the knob settings parsed out of the case names

    :param Path dataset_file: benchmark_dataset.csv to read
    :return: pandas DataFrame of the four_node runs
    """
    dataset = pd.read_csv(dataset_file)
    dataset = dataset[dataset["case_name"].str.startswith("four_node")].copy()

    parsed = pd.DataFrame(
        [_parse(name) for name in dataset["case_name"]], index=dataset.index
    )
    dataset = pd.concat([dataset, parsed], axis=1)

    # 0 threads means every core of the machine, which on this server is 48
    dataset["requested"] = dataset["gurobi_threads"].replace(0, 48)
    dataset["kind"] = np.where(dataset["n_binvars"] == 0, "pure LP", "MIP")
    dataset["config"] = (
        dataset["case_name"]
        .str.replace(r"_thr\d+", "", regex=True)
        .str.replace(r"_aff\d+", "", regex=True)
    )

    return dataset


def paired(dataset: pd.DataFrame, metric: str):
    """
    Builds the ratios of runs that differ in exactly one knob.

    Pairing divides the size of the model out, which is the only way to see what
    a modelling choice costs rather than what the model it produces costs.

    :param dataset: DataFrame returned by load
    :param str metric: column to take the ratio of
    :return: DataFrame with one row per pair
    """
    rows = []

    for knob in KNOBS:
        keys = [other for other in KNOBS if other != knob] + ["td", "thr", "aff"]
        for _, group in dataset.groupby(keys, dropna=False):
            cheap = group[group[knob] != EXPENSIVE[knob]]
            dear = group[group[knob] == EXPENSIVE[knob]]
            if cheap.empty or dear.empty:
                continue

            low, high = cheap.iloc[0], dear.iloc[0]
            # A run stopped by the time limit has a runtime set by the limit,
            # so it cannot stand in a ratio
            if "optimal" not in (
                low["termination_condition"],
                high["termination_condition"],
            ):
                continue
            if (
                low["termination_condition"] != "optimal"
                or high["termination_condition"] != "optimal"
            ):
                continue
            if low[metric] <= 0:
                continue

            rows.append({"knob": knob, "ratio": high[metric] / low[metric]})

    return pd.DataFrame(rows)


def plot_cores(dataset: pd.DataFrame, output: Path):
    """
    Draws cores actually used against cores asked for

    :param dataset: DataFrame returned by load
    :param Path output: png file to write
    """
    figure, axis = plt.subplots(figsize=(8.2, 5.6))
    limits = [0.8, 60]

    axis.plot(limits, limits, ls="--", color=DARK, lw=1.2, zorder=2)
    axis.annotate(
        "if gurobi used what it was given",
        xy=(9, 9),
        xytext=(-10, 9),
        textcoords="offset points",
        rotation=44,
        fontsize=8.5,
        color=DARK,
        ha="right",
        rotation_mode="anchor",
    )

    # A little spread on the x axis, so that 150 runs at one thread count read
    # as a cloud rather than as a single column
    rng = np.random.default_rng(0)
    for kind, colour, marker, size in [
        ("MIP", BLUE, "o", 20),
        ("pure LP", ORANGE, "^", 34),
    ]:
        group = dataset[dataset["kind"] == kind]
        spread = group["requested"] * np.exp(rng.normal(0, 0.035, len(group)))
        axis.scatter(
            spread,
            group["parallelism_solve"],
            s=size,
            color=colour,
            alpha=0.5,
            marker=marker,
            edgecolors="none",
            zorder=3,
            label=f"{kind}, {len(group)} runs",
        )

    # The median has to be taken over configurations that exist at every thread
    # count, otherwise it compares 150 runs of every size at 4 and 48 threads
    # against a handful of large ones in between
    spans = dataset.groupby("config")["requested"].nunique()
    comparable = dataset[dataset["config"].isin(spans[spans >= 5].index)]
    middle = comparable.groupby("requested")["parallelism_solve"].median()
    axis.plot(
        middle.index,
        middle.values,
        color=RED,
        lw=2.4,
        marker="o",
        ms=6,
        zorder=5,
        label=f"median over the {comparable['config'].nunique()} configurations\n"
        "measured at every thread count",
    )

    ticks = sorted(dataset["requested"].unique())
    axis.set_xscale("log", base=2)
    axis.set_yscale("log", base=2)
    axis.set_xticks(ticks)
    axis.set_xticklabels(ticks)
    axis.set_yticks([1, 2, 4, 8, 16])
    axis.set_yticklabels([1, 2, 4, 8, 16])
    axis.set_xlim(*limits)
    axis.set_ylim(0.8, 20)
    axis.set_xlabel("solver threads requested")
    axis.set_ylabel("cores actually busy during the solve")
    axis.grid(alpha=0.22, which="both", lw=0.5)
    axis.legend(fontsize=9, frameon=False, loc="upper left")
    axis.annotate(
        "past 8 threads nothing changes:\nthe median sits at 2 to 3 cores",
        xy=(16, 2.6),
        xytext=(0, -64),
        textcoords="offset points",
        ha="center",
        fontsize=9.5,
        color=RED,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.1),
    )
    for requested, group in dataset.groupby("requested"):
        axis.annotate(
            f"n={len(group)}",
            xy=(requested, 0.83),
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=GREY,
        )

    figure.suptitle(
        "Asking for more cores does not make gurobi use them", fontsize=13, y=0.97
    )
    axis.set_title(
        "only the pure-LP runs, where the concurrent method fires, ever go high",
        fontsize=9.5,
        color=GREY,
    )
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    print(f"Wrote {output}")


def plot_memory(dataset: pd.DataFrame, output: Path):
    """
    Draws peak memory against model size, with the band a request should use

    :param dataset: DataFrame returned by load
    :param Path output: png file to write
    """
    x = np.log10(dataset["n_vars"])
    y = np.log10(dataset["rss_peak_os_mb"])
    slope, intercept = np.polyfit(x, y, 1)
    r_squared = np.corrcoef(x, y)[0, 1] ** 2
    residual = y - (intercept + slope * x)
    low, high = 10 ** residual.quantile(0.1), 10 ** residual.quantile(0.9)

    figure, axis = plt.subplots(figsize=(8.6, 5.6))
    grid = np.logspace(
        np.log10(dataset["n_vars"].min() * 0.8),
        np.log10(dataset["n_vars"].max() * 1.3),
        100,
    )
    fit = 10**intercept * grid**slope

    axis.fill_between(
        grid,
        fit * low,
        fit * high,
        color=BLUE,
        alpha=0.15,
        zorder=1,
        label=f"80 % of runs, x{low:.2f} to x{high:.2f}",
    )
    axis.plot(
        grid,
        fit,
        color=BLUE,
        lw=2,
        zorder=3,
        label=f"fit: $n^{{{slope:.2f}}}$, $R^2$ = {r_squared:.2f}",
    )
    axis.plot(
        grid,
        16 * grid / 1000,
        color=DARK,
        ls="--",
        lw=1.3,
        zorder=4,
        label="rule of thumb: 16 MB per 1000 variables",
    )

    for requested, colour, marker in [(48, ORANGE, "s"), (4, GREEN, "o")]:
        group = dataset[dataset["requested"] == requested]
        axis.scatter(
            group["n_vars"],
            group["rss_peak_os_mb"],
            s=17,
            color=colour,
            alpha=0.55,
            marker=marker,
            edgecolors="none",
            zorder=5,
            label=f"{requested} threads",
        )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("variables in the model")
    axis.set_ylabel("peak memory of the whole process [MB]")
    axis.grid(alpha=0.22, which="both", lw=0.5)
    axis.legend(fontsize=9, frameon=False, loc="upper left")
    axis.annotate(
        "the linear rule runs above the fit past ~1e5 variables:\n"
        "safe for a request, but it over-asks by 3x at 3 million",
        xy=(6e5, 4.5e4),
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=DARK,
    )

    figure.suptitle(
        "Memory can be sized from the model, to within a factor 2",
        fontsize=13,
        y=0.97,
    )
    axis.set_title(
        "this is what a Snellius --mem request should be built on",
        fontsize=9.5,
        color=GREY,
    )
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    print(f"Wrote {output}")


def plot_knobs(dataset: pd.DataFrame, output: Path):
    """
    Draws what each knob costs, against the noise floor

    :param dataset: DataFrame returned by load
    :param Path output: png file to write
    """
    titles = {
        "gurobi_runtime_s": (
            "What a knob costs in TIME",
            "only storage clears the noise",
        ),
        "rss_peak_os_mb": ("What a knob costs in MEMORY", "here the measurement works"),
    }

    figure, axes = plt.subplots(1, 2, figsize=(13.4, 5.4))
    order = None

    for axis, metric in zip(axes, ["gurobi_runtime_s", "rss_peak_os_mb"]):
        ratios = paired(dataset, metric)
        medians = ratios.groupby("knob")["ratio"].median().sort_values()
        if order is None:
            order = list(medians.index)

        noise = NOISE[metric]
        axis.axhspan(1 / noise, noise, color=RED, alpha=0.12, zorder=0)
        axis.axhline(1, color=DARK, lw=1, ls="--", zorder=1)

        boxes = axis.boxplot(
            [ratios[ratios["knob"] == knob]["ratio"].values for knob in order],
            patch_artist=True,
            widths=0.55,
            medianprops=dict(color="k", lw=1.7),
            flierprops=dict(marker="o", ms=2.6, mfc=GREY, mec="none"),
        )
        for patch, knob in zip(boxes["boxes"], order):
            clears = medians[knob] > noise or medians[knob] < 1 / noise
            patch.set_facecolor(GREEN if clears else GREY)
            patch.set_alpha(0.7)
            patch.set_edgecolor("k")

        axis.set_yscale("log")
        axis.set_xticklabels(
            [f"{LABELS[knob]}\n{medians[knob]:.2f}x" for knob in order],
            fontsize=8.6,
            rotation=28,
            ha="right",
        )
        axis.set_ylabel("expensive setting / cheap setting")
        axis.grid(alpha=0.22, axis="y", which="both", lw=0.5)
        title, verdict = titles[metric]
        axis.set_title(f"{title}\n{verdict}", fontsize=11)
        axis.annotate(
            f"shaded: run-to-run noise, x{noise:g}",
            xy=(0.03, 0.03),
            xycoords="axes fraction",
            ha="left",
            va="bottom",
            fontsize=8.5,
            color=RED,
        )

    axes[0].set_ylim(0.2, 2000)
    axes[0].set_yticks([0.25, 1, 10, 100, 1000])
    axes[0].set_yticklabels(["0.25x", "1x", "10x", "100x", "1000x"])
    axes[1].set_ylim(0.7, 8)
    axes[1].set_yticks([0.8, 1, 1.5, 2, 3, 5, 8])
    axes[1].set_yticklabels(["0.8x", "1x", "1.5x", "2x", "3x", "5x", "8x"])

    figure.suptitle(
        "Only storage survives the noise as a time cost, but every knob is "
        "measurable as a memory cost",
        fontsize=12.5,
        y=0.99,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output, dpi=160)
    print(f"Wrote {output}")


def main():
    """
    Command line interface of the resource guidance figures
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_FILE)
    parser.add_argument("--figures", type=Path, default=FIGURES_PATH)
    args = parser.parse_args()

    dataset = load(args.dataset)
    print(f"{len(dataset)} four_node runs")

    args.figures.mkdir(parents=True, exist_ok=True)
    plot_cores(dataset, args.figures / "four_node_cores_requested_vs_used.png")
    plot_memory(dataset, args.figures / "four_node_memory_predictor.png")
    plot_knobs(dataset, args.figures / "four_node_what_survives_the_noise.png")


if __name__ == "__main__":
    main()
