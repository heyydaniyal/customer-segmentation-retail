"""
Plotting helpers used across the notebooks.

We keep matplotlib defaults conservative and rely on the caller to pass an
Axes object when finer layout control is needed. Anything fancier than a
simple chart belongs in the notebook, not here.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram

from . import config as cfg


# ---------------------------------------------------------------------------
# Generic
# ---------------------------------------------------------------------------
def set_style():
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.figsize"] = cfg.FIGSIZE_DEFAULT
    plt.rcParams["axes.titleweight"] = "bold"


# ---------------------------------------------------------------------------
# Choosing K
# ---------------------------------------------------------------------------
def plot_k_metrics(metrics: pd.DataFrame, axes=None):
    """
    Plot the four K-choice metrics as a 2x2 grid of panels. `metrics` is the
    DataFrame returned by evaluate_k_range.

    `axes` must be a 2x2 array of Axes (as returned by plt.subplots(2, 2)),
    because this draws four panels. Leave it None to have one created. The
    parameter used to be called `ax` and was documented as if a single Axes
    would do, which raised AttributeError for every caller who tried it.
    """
    if axes is None:
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes = np.asarray(axes)
    if axes.size != 4:
        raise ValueError(
            f"plot_k_metrics draws four panels and needs four Axes, "
            f"got {axes.size}. Pass plt.subplots(2, 2)[1] or leave axes=None."
        )

    metric_specs = [
        ("inertia", "Inertia (lower is better, look for elbow)"),
        ("silhouette", "Silhouette (higher is better)"),
        ("davies_bouldin", "Davies-Bouldin (lower is better)"),
        ("calinski_harabasz", "Calinski-Harabasz (higher is better)"),
    ]

    for ax_, (col, title) in zip(axes.flatten(), metric_specs):
        ax_.plot(metrics.index, metrics[col], marker="o")
        ax_.set_title(title)
        ax_.set_xlabel("k")
        ax_.set_ylabel(col)

    plt.tight_layout()
    return axes


# ---------------------------------------------------------------------------
# Cluster profiles
# ---------------------------------------------------------------------------
def plot_cluster_heatmap(profile: pd.DataFrame, title: str = "", ax=None):
    """
    Heatmap of cluster means relative to overall mean (output of
    clustering.cluster_profile). 100 = the population average; values far
    from 100 indicate distinguishing features.

    Rows for near-zero-mean features carry a different scale (standardised
    difference, also centred on 100) because a ratio-of-mean is meaningless
    when the denominator is ~0. cluster_profile records those in
    `.attrs["zscored_features"]`; we suffix their labels with " (z)" and say
    so on the colourbar rather than letting two scales share one legend
    silently.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, 1.4 * profile.shape[1]), max(4, 0.4 * profile.shape[0])))

    zscored = profile.attrs.get("zscored_features", [])
    plotted = profile.rename(index={f: f"{f} (z)" for f in zscored})
    label = ("100 = population average\n(% of mean; rows marked (z) are "
             "standardised difference)") if zscored else "% of overall mean (100 = average)"

    sns.heatmap(
        plotted,
        center=100,
        cmap="RdBu_r",
        annot=True,
        fmt=".0f",
        cbar_kws={"label": label},
        ax=ax,
    )
    ax.set_title(title)
    return ax


# ---------------------------------------------------------------------------
# Dimensionality reduction for visual sanity-check
# ---------------------------------------------------------------------------
def tsne_scatter(
    X: np.ndarray,
    labels: np.ndarray,
    sample_size: int = 5000,
    random_state: int = cfg.RANDOM_STATE,
    title: str = "t-SNE projection",
    ax=None,
):
    """
    Project X to 2D with t-SNE and scatter colored by labels.

    t-SNE on 33k customers is slow, so we subsample. This is for visual
    inspection only, never for clustering itself.
    """
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X), size=min(sample_size, len(X)), replace=False)

    proj = TSNE(
        n_components=2,
        random_state=random_state,
        perplexity=30,
        init="pca",
        learning_rate="auto",
    ).fit_transform(X[idx])

    if ax is None:
        fig, ax = plt.subplots(figsize=cfg.FIGSIZE_DEFAULT)

    sns.scatterplot(
        x=proj[:, 0],
        y=proj[:, 1],
        hue=labels[idx],
        palette="tab10",
        s=14,
        alpha=0.7,
        ax=ax,
        legend="full",
    )
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    return ax, proj, idx


# ---------------------------------------------------------------------------
# Dendrogram
# ---------------------------------------------------------------------------
def plot_dendrogram(linkage_matrix, truncate_p: int = 30, ax=None, title: str = ""):
    """Dendrogram of an Agglomerative clustering, truncated for readability."""
    if ax is None:
        fig, ax = plt.subplots(figsize=cfg.FIGSIZE_WIDE)
    dendrogram(
        linkage_matrix,
        truncate_mode="lastp",
        p=truncate_p,
        leaf_rotation=90,
        leaf_font_size=10,
        show_contracted=True,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("cluster size or sample index")
    ax.set_ylabel("distance")
    return ax
