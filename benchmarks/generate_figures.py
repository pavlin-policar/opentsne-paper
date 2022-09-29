import os
import re
from dataclasses import dataclass
from functools import wraps, cmp_to_key
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

matplotlib.rcParams["pdf.fonttype"] = 42  # Make fonts editable in AI


BENCHMARK_DIR = "../benchmarks/logs"


def make_regex_func(pattern):
    def _wrapper(fname):
        matches = [re.findall(pattern, line) for line in open(fname)]
        matches = list(filter(len, matches))
        matches = list(map(lambda x: float(x[0]), matches))

        return np.array(matches)[1:]  # skip first -- warmup

    return _wrapper


def from_minutes(f):
    @wraps(f)
    def _wrapper(*args, **kwargs):
        return f(*args, **kwargs) * 60
    return _wrapper


@dataclass
class Spec:
    name: str
    log_name: str
    cores: int
    lang: str
    parser: Callable

    @property
    def display_name(self):
        return f"{self.name} ({self.cores} {'cores' if self.cores > 1 else 'core'})"

    def __hash__(self):
        return hash(self.display_name)

    def __eq__(self, other):
        return self.display_name == other.display_name

    def __gt__(self, other):
        return self.display_name > other.display_name


def get_benchmarks(spec: Spec, directory: str) -> List[Dict]:
    # Create regex pattern to match file names
    pattern = r"(.+)" + f"--{spec.log_name}--{spec.cores}_core--" + r"(\d+)_samples.log"

    data = []
    for f in os.listdir(directory):
        if match := re.findall(pattern, f):
            dataset = match[0][0]
            n_samples = int(match[0][1])

            times = spec.parser(os.path.join(directory, f))
            for t in times:
                data.append({
                    "Spec": spec,
                    "Name": spec.display_name,
                    "Num. Samples": n_samples,
                    "Time (sec)": t,
                    "Time (min)": t / 60,
                    "Benchmark": dataset,
                })

    return data


# DEFINE PARSERS
# openTSNE
parse_opentsne = make_regex_func(r"openTSNE: Full (\d+\.\d+)")
# FIt-SNE
parse_fitsne = make_regex_func(r"FItSNE: (\d+\.\d+)")
# MulticoreTSNE
parse_multicore = make_regex_func(r"MulticoreTSNE: (\d+\.\d+)")
# scikit-learn
parse_sklearn = make_regex_func(r"sklearn: (\d+\.\d+)")
# UMAP
parse_umap = make_regex_func(r"UMAP: (\d+\.\d+)")
# Rtsne
parse_rtsne = make_regex_func(r"Rtsne benchmark time: (\d+\.\d+)")
# TSne.jl
parse_jl = make_regex_func(r"(\d+\.\d+) seconds ")

# Define implementations
implementations = [
    # openTSNE
    Spec("openTSNE FFT", log_name="openTSNEFFT", cores=1, lang="Python", parser=parse_opentsne),
    Spec("openTSNE FFT", log_name="openTSNEFFT", cores=8, lang="Python", parser=parse_opentsne),
    Spec("openTSNE BH", log_name="openTSNEBH", cores=1, lang="Python", parser=parse_opentsne),
    Spec("openTSNE BH", log_name="openTSNEBH", cores=8, lang="Python", parser=parse_opentsne),

    # FIt-SNE
    Spec("FIt-SNE", log_name="FItSNE", cores=1, lang="Python", parser=parse_fitsne),
    Spec("FIt-SNE", log_name="FItSNE", cores=8, lang="Python", parser=parse_fitsne),

    # MulticoreTSNE
    Spec("MulticoreTSNE", log_name="MulticoreTSNE", cores=1, lang="Python", parser=parse_multicore),
    Spec("MulticoreTSNE", log_name="MulticoreTSNE", cores=8, lang="Python", parser=parse_multicore),

    # scikit-learn
    Spec("scikit-learn", log_name="sklearn", cores=1, lang="Python", parser=parse_sklearn),
    Spec("scikit-learn", log_name="sklearn", cores=8, lang="Python", parser=parse_sklearn),

    # UMAP
    Spec("UMAP", log_name="UMAP", cores=1, lang="Python", parser=parse_umap),
    Spec("UMAP", log_name="UMAP", cores=8, lang="Python", parser=parse_umap),

    # Rtsne
    Spec("Rtsne", log_name="Rtsne", cores=1, lang="R", parser=parse_rtsne),
    Spec("Rtsne", log_name="Rtsne", cores=8, lang="R", parser=parse_rtsne),

    # TSne.jl
    Spec("TSne.jl", log_name="TSne-jl", cores=1, lang="Julia", parser=parse_jl),
]
impl_mapping = {impl.name: impl for impl in implementations}

# Read data into dataframe
data = []
for impl in implementations:
    data.extend(get_benchmarks(impl, BENCHMARK_DIR))
data = pd.DataFrame(data)


def generate_benchmark_plot(colors, title, fname):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.despine(offset=20)

    ax.set_title(
        title,
        loc="left",
        fontdict={"fontsize": "13"},
        pad=15,
    )
    ax.set_xlabel("Dataset size [samples]")
    ax.set_ylabel("Time [min]")

    ax.grid(color="0.9", linestyle="--", linewidth=1)

    agg = data.groupby(["Spec", "Name", "Num. Samples"])["Time (min)"].aggregate(["mean", "std"]).reset_index()

    # Sort the labels, so they appear in the correct order in the legend
    color_names = np.array(list(colors.keys()))
    color_name_idx = {v: k for k, v in enumerate(color_names)}

    def sort_cmp(impl1, impl2):
        impl1_idx = color_name_idx[impl1.name]
        impl2_idx = color_name_idx[impl2.name]
        return impl1_idx - impl2_idx

    # Skip the ones that don't belong on the plot
    order = [impl for impl in agg["Spec"].unique() if impl.name in colors]

    for impl in sorted(order, key=cmp_to_key(sort_cmp)):
        subset = agg.query(f"Name == '{impl.display_name}'")

        ax.plot(
            subset["Num. Samples"],
            subset["mean"],
            label=impl.display_name,
            c=colors[impl.name],
            linestyle="solid" if impl.cores == 1 else "dashed",
        )
        ax.fill_between(
            subset["Num. Samples"],
            subset["mean"] + subset["std"],
            subset["mean"] - subset["std"],
            alpha=0.1,
            color=colors[impl.name],
        )

    ax.set_xlim(0, data["Num. Samples"].max())

    if data["Benchmark"].unique()[0] == "macosko_2015":  # small benchmark
        ax.set_ylim(0, 10)
    else:
        ax.set_ylim(0, 130)
        ax.set_yticks(range(0, 130, 15))

    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda x, p: format(int(x), ",").replace(",", "."))
    )

    # The legend will only indicate solid/dashed lines for core count
    # The line labels for the different implementations will be added later in AI
    legend_patches = [
        Line2D([0], [0], color="#000", lw=2, linestyle="solid"),
        Line2D([0], [0], color="#000", lw=2, linestyle="dashed"),
    ]
    ax.legend(
        legend_patches,
        ["1 core", "8 cores"],
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.99),
        bbox_transform=ax.transAxes,
        ncol=1,
    )

    # BETTER/WORSE indicator text
    text_kwargs = dict(
        color="#BBBBBB",
        fontsize=16,
        fontstretch="expanded",
        fontweight="bold",
        transform=ax.transAxes,
    )
    hoff = 0.04
    voff = 0.04
    ax.text(1 - hoff, 0 + voff, "Better", ha="right", va="center", **text_kwargs)
    ax.text(0 + hoff, 1 - voff, "Worse", ha="left", va="center", **text_kwargs)

    plt.tight_layout()
    plt.savefig(
        fname,
        bbox_inches="tight",
        dpi=160,
    )

    return fig, ax


# PYTHON COMPARISON
colors = {
    "openTSNE FFT": "#4C72B0",
    "FIt-SNE": "#DD8452",
    "MulticoreTSNE": "#55A868",
    "scikit-learn": "#C44E52",
}
generate_benchmark_plot(
    colors,
    title="Comparison with other Python implementations",
    fname=os.path.join("..", "paper", "benchmarks_python.pdf"),
)
plt.show()


# PROGRAMMING LANGUAGES COMPARISON
colors = {
    "openTSNE FFT": "#4C72B0",
    "Rtsne": "#DD8452",
    "TSne.jl": "#55A868",
    # "MulticoreTSNE": "#C44E52",
}
generate_benchmark_plot(
    colors,
    title="Comparison with other programming languages",
    fname=os.path.join("..", "paper", "benchmarks_langs.pdf"),
)
plt.show()
