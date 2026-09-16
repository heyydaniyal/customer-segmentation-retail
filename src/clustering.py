"""
Clustering utilities.

The interface is small. The notebook should be able to do:

    scaler, X = scale_perspective(df, features)
    metrics = evaluate_k_range(X, k_range=range(2, 9))
    kmeans, labels = fit_kmeans(X, n_clusters=4)

This module also handles:
  - content-based cluster naming (so cluster IDs don't matter for
    downstream logic)
  - model persistence (so a new customer can be classified without
    re-fitting)
  - stability analysis across random seeds (defends against the concern
    that re-running could shuffle cluster identities)
"""

from __future__ import annotations

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
)
from sklearn.preprocessing import StandardScaler

from . import config as cfg


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------
def scale_perspective(df, features):
    """Standardise the features for one clustering perspective."""
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features].values)
    return scaler, X


# ---------------------------------------------------------------------------
# Evaluating K
# ---------------------------------------------------------------------------
N_INIT = 20  # shared by evaluate_k_range and fit_kmeans


def evaluate_k_range(X, k_range=range(2, 9), random_state=cfg.RANDOM_STATE,
                     sample_size=5000, n_init=N_INIT):
    """
    Fit K-Means for each K and compute four quality metrics.

    Uses the same `n_init` as fit_kmeans, so the metrics reported for the
    chosen K describe the model that actually gets shipped. (They previously
    differed, 10 here against 20 there — on this data the partitions are
    identical either way, but quoting a score from a model you did not fit is
    the kind of small inconsistency that invites a bigger question.)

    Silhouette is computed on a fixed random subsample because it is O(n^2) in
    memory; Davies-Bouldin and Calinski-Harabasz are O(n) and use every row.
    The subsample is drawn once, outside the loop, so all K values are scored
    on exactly the same customers and the curve is comparable across K.
    """
    rows = []
    rng = np.random.default_rng(random_state)
    if len(X) > sample_size:
        sample_idx = rng.choice(len(X), size=sample_size, replace=False)
    else:
        sample_idx = np.arange(len(X))

    for k in k_range:
        km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        labels = km.fit_predict(X)
        rows.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(X[sample_idx], labels[sample_idx]),
            "davies_bouldin": davies_bouldin_score(X, labels),
            "calinski_harabasz": calinski_harabasz_score(X, labels),
        })
    return pd.DataFrame(rows).set_index("k")


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
def fit_kmeans(X, n_clusters, random_state=cfg.RANDOM_STATE, n_init=N_INIT):
    """Fit K-Means with sensible defaults and return model + labels.

    Same `n_init` as evaluate_k_range, so the model shipped is the model
    scored."""
    km = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state)
    labels = km.fit_predict(X)
    return km, labels


def hierarchical_validation(X, n_clusters, method="ward", sample_size=5000,
                             random_state=cfg.RANDOM_STATE):
    """Run Agglomerative clustering on a sample. Returns linkage matrix +
    silhouette so the notebook can plot a dendrogram."""
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X), size=min(sample_size, len(X)), replace=False)
    X_sample = X[idx]
    Z = linkage(X_sample, method=method)
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    return {
        "linkage_matrix": Z,
        "sample_indices": idx,
        "sample_labels": labels,
        "silhouette": silhouette_score(X_sample, labels),
    }


# ---------------------------------------------------------------------------
# Stability analysis
# ---------------------------------------------------------------------------
def stability_across_seeds(X, n_clusters, seeds=(0, 1, 2, 17, 42, 99)):
    """
    Re-fit K-Means with different random seeds and compute pairwise ARI.

    A high ARI across seeds means the clusters are stable: the partition
    of customers is the same even if the integer IDs shuffle. This is the
    proper defence against the "could rerun and get different labels"
    concern.

    Returns the mean and standard deviation of pairwise ARI, and the matrix.
    """
    label_sets = []
    for seed in seeds:
        km = KMeans(n_clusters=n_clusters, n_init=N_INIT, random_state=seed)
        label_sets.append(km.fit_predict(X))

    n = len(seeds)
    aris = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            aris[i, j] = adjusted_rand_score(label_sets[i], label_sets[j])

    # Off-diagonal entries
    mask = ~np.eye(n, dtype=bool)
    return {
        "ari_matrix": pd.DataFrame(aris, index=list(seeds), columns=list(seeds)),
        "mean_ari": aris[mask].mean(),
        "std_ari": aris[mask].std(),
        "min_ari": aris[mask].min(),
    }


# ---------------------------------------------------------------------------
# Profiling clusters
# ---------------------------------------------------------------------------
def cluster_profile(df, labels, features, label_name="cluster",
                    zero_centered_threshold=0.2):
    """
    Compare cluster means against the overall mean.

    Returns a DataFrame: rows are features, columns are clusters, values
    are the cluster mean expressed as a percentage of the overall mean.

    Special handling for zero-centered features (sin/cos cyclic encodings,
    standardized z-scores): for these the overall mean is at or near zero
    and the percentage-of-mean ratio explodes into meaningless artifacts
    (e.g. -400% just because the denominator is +0.05). For features whose
    absolute overall mean is below `zero_centered_threshold` times the
    feature's standard deviation we instead report
       100 + (cluster_mean - overall_mean) / overall_std * 100
    which centres at 100, uses the same colour scale as the ratio columns,
    and reflects how many standard deviations the cluster sits from the
    overall mean. The result reads on the same heatmap without producing
    spurious extreme values.

    Because two different scales can appear in one table, the names of the
    z-scored features are recorded in `result.attrs["zscored_features"]`.
    plot_cluster_heatmap reads that and marks those rows, so a reader is
    never left guessing which scale a row is on. In this dataset only
    `typical_hour_sin` ever qualifies.

    Implementation note: `near_zero` is indexed by feature name while the
    intermediate frames are indexed by cluster id. `DataFrame.where` aligns a
    Series condition on the *index*, and ignores `axis=`, so passing the
    Series directly silently yields an all-NaN condition -> every cell takes
    the `other` branch. The mask is therefore broadcast to full frame shape
    explicitly. (This was a live bug: it forced every feature onto the
    standardised-difference scale while the colourbar still said "% of
    overall mean", and produced negative "percentages of a mean" on the
    product-mix heatmap.)
    """
    out = df[features].copy()
    out[label_name] = labels
    overall_mean = out[features].mean()
    overall_std = out[features].std().replace(0, 1.0)  # avoid 0-division
    grouped = out.groupby(label_name)[features].mean()

    # Identify near-zero-mean features (cyclic encodings, z-scored inputs)
    near_zero = overall_mean.abs() < zero_centered_threshold * overall_std

    # Ratio-of-mean for normal features
    ratio = (grouped / overall_mean.replace(0, np.nan)) * 100
    # Standardised-difference (centred at 100) for near-zero-mean features
    std_diff = ((grouped - overall_mean) / overall_std) * 100 + 100

    # Broadcast the per-feature mask across cluster rows so the condition has
    # the same shape as the frames it selects between.
    mask = pd.DataFrame(
        np.tile(near_zero.to_numpy(), (len(grouped), 1)),
        index=grouped.index,
        columns=grouped.columns,
    )

    relative = ratio.where(~mask, std_diff).round(1).T
    relative.columns = [f"cluster_{c}" for c in relative.columns]
    relative.attrs["zscored_features"] = [f for f in features if near_zero[f]]
    return relative


def cluster_sizes(labels):
    """Count of customers per cluster, sorted by cluster id."""
    return pd.Series(labels).value_counts().sort_index()


# ---------------------------------------------------------------------------
# Alternative algorithms (for comparison / rejection with reasons)
# ---------------------------------------------------------------------------
def k_distance_curve(X, k=None):
    """
    Return the sorted distance to each point's k-th nearest neighbour.

    Plotting this curve is the standard, non-arbitrary way to choose DBSCAN's
    eps: the "knee" of the curve (where it turns sharply upward) is the eps
    where points start to become isolated. minPts defaults to 2*dimensions,
    a common rule of thumb.
    """
    from sklearn.neighbors import NearestNeighbors
    if k is None:
        k = 2 * X.shape[1]
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    dists, _ = nn.kneighbors(X)
    return np.sort(dists[:, -1]), k


def dbscan_scan(X, eps_values, min_samples=None):
    """
    Run DBSCAN at several eps values and report cluster count + noise fraction.

    Returns a DataFrame so the notebook can show how DBSCAN behaves across the
    plausible eps range read off the k-distance curve. On dense, single-blob
    data DBSCAN typically returns one cluster (everything is reachable) or
    shatters into noise, neither of which is a useful customer segmentation.
    """
    from sklearn.cluster import DBSCAN
    if min_samples is None:
        min_samples = 2 * X.shape[1]
    rows = []
    for eps in eps_values:
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise = float((labels == -1).mean())
        rows.append({"eps": eps, "min_samples": min_samples,
                     "n_clusters": n_clusters, "noise_fraction": noise})
    return pd.DataFrame(rows)


def meanshift_summary(X, sample_size=4000, quantile=0.2,
                      random_state=cfg.RANDOM_STATE):
    """
    Run Mean-Shift on a sample and summarise the result.

    Mean-Shift is O(n^2) per iteration, so we sample. We report the number of
    clusters found and the share of points in the largest cluster. On this
    data Mean-Shift either collapses to a single mode (value perspective) or
    produces one dominant blob plus many tiny fragments (mix perspective) —
    again not a usable segmentation, which is the point of showing it.
    """
    from sklearn.cluster import MeanShift, estimate_bandwidth
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X), size=min(sample_size, len(X)), replace=False)
    Xs = X[idx]
    bandwidth = estimate_bandwidth(Xs, quantile=quantile, random_state=random_state)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(Xs)
    sizes = pd.Series(ms.labels_).value_counts()
    return {
        "bandwidth": float(bandwidth),
        "n_clusters": int(len(sizes)),
        "largest_cluster_fraction": float(sizes.iloc[0] / len(Xs)),
        "sample_size": len(Xs),
    }


# ---------------------------------------------------------------------------
# Content-based cluster naming
# ---------------------------------------------------------------------------
def name_value_clusters(df, label_col="value_cluster"):
    """
    Map raw value-cluster IDs to content-based names.

    Looks at the *centroid* of each cluster, not the integer ID, so even
    if K-Means shuffles IDs across runs the names stay consistent.

    Naming note: this axis is defined by basket BREADTH and ENGAGEMENT
    (lifetime_total_distinct_products, distinct_stores_visited, tenure),
    NOT by total spend. We deliberately avoid the label "high value"
    because the engaged cluster does not always have the highest mean
    spend, and we deliberately use "routine" rather than "low engagement"
    or "low value" because this cluster's mean spend is roughly the
    population average (~85% of overall), not low - what makes it distinct
    is the *narrow product breadth* (about half the distinct-products count
    of the engaged cluster) and the very low promo usage. These customers
    shop in a focused, habitual way rather than disengaged from the brand.

    Returns: dict {cluster_id -> name}
    Names used: 'engaged', 'promo_hunter', 'routine'
    """
    expected = 3
    centroids = df.groupby(label_col)[
        ["total_spend_log", "lifetime_total_distinct_products", "promo_pct"]
    ].mean()
    # A raise, not an assert: `python -O` strips asserts, and this guard is
    # the only thing standing between a changed K and silently mislabelled
    # segments.
    if len(centroids) != expected:
        raise ValueError(
            f"name_value_clusters expects exactly {expected} clusters, "
            f"got {len(centroids)}. If K changed, update this function."
        )

    mapping = {}
    # Promo hunter: highest promo_pct
    promo_hunter_id = centroids["promo_pct"].idxmax()
    mapping[promo_hunter_id] = "promo_hunter"

    # Of the remaining two, the higher distinct_products one is the
    # engaged (broad-basket) cluster; the other is routine (narrow basket,
    # average spend, low promo).
    remaining = [c for c in centroids.index if c != promo_hunter_id]
    sorted_remaining = sorted(
        remaining,
        key=lambda c: centroids.loc[c, "lifetime_total_distinct_products"],
        reverse=True,
    )
    mapping[sorted_remaining[0]] = "engaged"
    mapping[sorted_remaining[1]] = "routine"
    return mapping


def name_mix_clusters(df, label_col="mix_cluster"):
    """
    Map raw mix-cluster IDs to content-based names.

    Each cluster gets a name based on which category share it most
    over-indexes on relative to the overall mean.

    Naming note: the cluster that over-indexes on vegetables and hygiene
    in the spend shares is labelled 'young_parents' rather than 'healthy'.
    The basket-level association rules for these customers (notebook 04)
    are dominated by babies food, napkins and cooking oil, and the spend
    profile (vegetables, hygiene, non-alcohol drinks) is consistent with a
    household with small children. 'young_parents' reconciles both views;
    'healthy' would only reflect the spend shares and contradict the
    basket evidence.

    IMPORTANT - this function is *constructive*, not diagnostic. It assigns
    each of the four names exactly once, greedily, whatever the centroids
    look like. It can therefore never fail to "recover the four clusters",
    and running it on a re-clustered partition is not evidence that the two
    partitions agree. Use adjusted_rand_score for that.

    Returns: dict {cluster_id -> name}
    Names: 'tech', 'family_provisioner', 'young_parents', 'grocery_generalist'
    """
    expected = 4
    share_cols = [c for c in df.columns if c.startswith("share_")]
    centroids = df.groupby(label_col)[share_cols].mean()
    if len(centroids) != expected:
        raise ValueError(
            f"name_mix_clusters expects exactly {expected} clusters, "
            f"got {len(centroids)}. If K changed, update this function."
        )
    overall = df[share_cols].mean()
    relative = centroids / overall  # ratio per share

    mapping = {}
    available = list(centroids.index)

    # Tech: most extreme on share_electronics or share_videogames
    tech_score = relative[["share_electronics", "share_videogames"]].max(axis=1)
    tech_id = tech_score.idxmax()
    mapping[tech_id] = "tech"
    available.remove(tech_id)

    # Family provisioner: most extreme on share_meat or share_petfood
    fp_score = relative.loc[available, ["share_meat", "share_petfood"]].max(axis=1)
    fp_id = fp_score.idxmax()
    mapping[fp_id] = "family_provisioner"
    available.remove(fp_id)

    # Young parents: most extreme on share_vegetables (with hygiene/babies
    # corroboration in the basket data)
    yp_id = relative.loc[available, "share_vegetables"].idxmax()
    mapping[yp_id] = "young_parents"
    available.remove(yp_id)

    # Remainder is grocery-generalist
    mapping[available[0]] = "grocery_generalist"
    return mapping


def name_demo_clusters(df, label_col="demo_cluster"):
    """Map demo-cluster IDs to a content-based name based on dominant degree."""
    degree_cols = ["degree_Bsc", "degree_Msc", "degree_Phd", "degree_None"]
    centroids = df.groupby(label_col)[degree_cols].mean()
    mapping = {}
    for cluster_id, row in centroids.iterrows():
        dominant_col = row.idxmax()
        dominant = dominant_col.replace("degree_", "").lower()
        mapping[cluster_id] = f"demo_{dominant}"
    return mapping


# ---------------------------------------------------------------------------
# Combining perspectives into a final segment - content-based
# ---------------------------------------------------------------------------
# NOTE ON METHOD: this lookup is a deliberate business-rule overlay on top of
# two independent clustering perspectives (value/engagement and product mix).
# It is NOT itself a clustering algorithm, and we do not claim it is. The two
# perspectives are learned by K-Means; this table is the documented,
# defensible mapping a marketing team would apply to turn those two axes into
# named segments. Two specific business decisions are encoded here:
#   1. All four (promo_hunter, *) cells collapse to a single "Promo Hunter"
#      segment, because for that tier promo-seeking behaviour dominates the
#      product-mix signal: a deal-driven shopper is best served by deal-driven
#      offers regardless of which categories they lean toward.
#   2. The value axis is engagement/breadth, not spend, so the labels say
#      "Engaged"/"Routine", never "High-Value"/"Low-Value".
FINAL_SEGMENT_LOOKUP = {
    ("engaged", "tech"):                 "Engaged Tech",
    ("engaged", "family_provisioner"):   "Engaged Family Provisioner",
    ("engaged", "young_parents"):        "Engaged Young Parents",
    ("engaged", "grocery_generalist"):   "Engaged Generalist",
    ("promo_hunter", "tech"):            "Promo Hunter",
    ("promo_hunter", "family_provisioner"): "Promo Hunter",
    ("promo_hunter", "young_parents"):   "Promo Hunter",
    ("promo_hunter", "grocery_generalist"): "Promo Hunter",
    ("routine", "tech"):                  "Routine Tech",
    ("routine", "family_provisioner"):    "Routine Family Provisioner",
    ("routine", "young_parents"):         "Routine Young Parents",
    ("routine", "grocery_generalist"):    "Routine Generalist",
}

# Segments smaller than this are folded into their parent generalist segment.
# Small segments cannot support real marketing campaigns; merging keeps the
# segmentation actionable.
MIN_SEGMENT_SIZE = 200


def learned_merge_labels(df, value_label_col="value_cluster",
                         mix_label_col="mix_cluster", n_final=7,
                         random_state=cfg.RANDOM_STATE):
    """
    Alternative to the business-rule lookup: a *learned* second-stage merge.

    We form the joint value x mix groups, compute each group's centroid in the
    standardised value+mix feature space, then run Ward agglomerative
    clustering on those centroids and cut to n_final groups. This makes the
    second stage a genuine clustering step rather than a hand-authored table.

    We provide this so the project can show both options and discuss the
    trade-off:
      - the lookup is more interpretable and lets us encode the deliberate
        "promo behaviour dominates for the promo tier" decision;
      - the learned merge is more algorithmic and avoids hand-authored rules,
        at the cost of names that have to be assigned to the merged groups
        after the fact.

    Returns a Series of integer merged-group labels aligned to df.
    """
    from scipy.cluster.hierarchy import linkage, fcluster

    value_feats = cfg.VALUE_FEATURES
    mix_feats = cfg.PRODUCT_MIX_FEATURES
    joint_feats = value_feats + mix_feats

    # Standardise the joint feature space once
    scaler = StandardScaler()
    Xj = scaler.fit_transform(df[joint_feats].values)
    Xj = pd.DataFrame(Xj, index=df.index, columns=joint_feats)

    group_key = (df[value_label_col].astype(str) + "_" +
                 df[mix_label_col].astype(str))
    centroids = Xj.groupby(group_key).mean()

    # Ward linkage on the joint cluster centroids, cut to n_final
    Z = linkage(centroids.values, method="ward")
    n_cut = min(n_final, len(centroids))
    centroid_labels = fcluster(Z, t=n_cut, criterion="maxclust")
    centroid_map = dict(zip(centroids.index, centroid_labels))

    return group_key.map(centroid_map).rename("merged_group")


def assign_final_segments(df, value_label_col="value_cluster",
                          mix_label_col="mix_cluster"):
    """
    Assign final named segments using content-based perspective names.

    1. Resolve each perspective's integer IDs to content-based names.
    2. Look up the final segment for each (value_name, mix_name) pair.
    3. Fold tiny segments (< MIN_SEGMENT_SIZE) into their generalist
       counterpart so every named segment is large enough to act on.

    Returns: (segments Series, naming info dict)
    """
    value_map = name_value_clusters(df, value_label_col)
    mix_map = name_mix_clusters(df, mix_label_col)

    value_named = df[value_label_col].map(value_map)
    mix_named = df[mix_label_col].map(mix_map)

    segments = pd.Series(
        [FINAL_SEGMENT_LOOKUP[(v, m)] for v, m in zip(value_named, mix_named)],
        index=df.index,
        name="segment",
    )

    counts = segments.value_counts()
    too_small = counts[counts < MIN_SEGMENT_SIZE].index.tolist()

    fold_map = {}
    for small in too_small:
        if small.startswith("Engaged"):
            fold_map[small] = "Engaged Generalist"
        elif small.startswith("Routine"):
            fold_map[small] = "Routine Generalist"

    if fold_map:
        segments = segments.replace(fold_map)

    return segments, {"value_map": value_map, "mix_map": mix_map,
                      "folded": fold_map}


# ---------------------------------------------------------------------------
# Persistence + inference
# ---------------------------------------------------------------------------
def save_models(scaler_v, km_v, scaler_p, km_p,
                value_map, mix_map, folded,
                preproc_params=None,
                scaler_d=None, km_d=None, demo_map=None,
                output_dir=None):
    """
    Save scalers, KMeans models, preprocessing params, and the
    integer-to-name mappings.

    `preproc_params` is the dict from preprocessing.fit_preprocessing_params:
    the frozen medians, gender mode and spend caps. Persisting it is what
    makes scoring independent of batch composition — without it, inference
    re-derives training statistics from whatever rows it happens to see.

    The demographic perspective is optional and saved only if provided.
    In this dataset the demographic clustering essentially recovers the
    `degree` column, so it is computed for descriptive context in the
    notebook but not used in the final segment label and not required for
    inference. The arguments are kept in the signature so older calling
    code does not break.

    Persisting the mappings alongside the models is what makes
    predict_segment robust to cluster ID shuffling: when we load, we know
    exactly which integer label corresponds to which named cluster.
    """
    output_dir = output_dir or cfg.OUTPUT_DIR
    models_dir = Path(output_dir) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler_v, models_dir / "scaler_value.joblib")
    joblib.dump(km_v, models_dir / "kmeans_value.joblib")
    joblib.dump(scaler_p, models_dir / "scaler_mix.joblib")
    joblib.dump(km_p, models_dir / "kmeans_mix.joblib")

    payload = {
        "value_map": {int(k): v for k, v in value_map.items()},
        "mix_map": {int(k): v for k, v in mix_map.items()},
        "folded": folded,
    }

    if preproc_params is not None:
        with open(models_dir / "preprocessing_params.json", "w") as f:
            json.dump(preproc_params, f, indent=2)

    if scaler_d is not None and km_d is not None and demo_map is not None:
        joblib.dump(scaler_d, models_dir / "scaler_demo.joblib")
        joblib.dump(km_d, models_dir / "kmeans_demo.joblib")
        payload["demo_map"] = {int(k): v for k, v in demo_map.items()}

    with open(models_dir / "cluster_naming.json", "w") as f:
        json.dump(payload, f, indent=2)

    return models_dir


def load_models(models_dir=None):
    """Load everything saved by save_models. Returns a dict."""
    models_dir = Path(models_dir or (cfg.OUTPUT_DIR / "models"))
    out = {
        "scaler_v": joblib.load(models_dir / "scaler_value.joblib"),
        "kmeans_v": joblib.load(models_dir / "kmeans_value.joblib"),
        "scaler_p": joblib.load(models_dir / "scaler_mix.joblib"),
        "kmeans_p": joblib.load(models_dir / "kmeans_mix.joblib"),
    }
    with open(models_dir / "cluster_naming.json") as f:
        payload = json.load(f)
    out["value_map"] = {int(k): v for k, v in payload["value_map"].items()}
    out["mix_map"] = {int(k): v for k, v in payload["mix_map"].items()}
    out["folded"] = payload["folded"]

    params_path = models_dir / "preprocessing_params.json"
    if params_path.exists():
        with open(params_path) as f:
            out["preproc_params"] = json.load(f)
    else:
        out["preproc_params"] = None

    return out


def predict_segment(customer_df, models=None):
    """
    Classify one or more *preprocessed* customers into final segments.

    customer_df must already contain all the engineered features (i.e. it
    is the output of build_features). Pass models=None to load from disk,
    or pass the dict from load_models() to avoid disk hits.

    Uses only the value and product-mix perspectives. The demographic
    perspective is not used because it essentially recovers the `degree`
    column and adds no information to the final segment.
    """
    if models is None:
        models = load_models()

    Xv = models["scaler_v"].transform(customer_df[cfg.VALUE_FEATURES].values)
    Xp = models["scaler_p"].transform(customer_df[cfg.PRODUCT_MIX_FEATURES].values)

    v_ids = models["kmeans_v"].predict(Xv)
    p_ids = models["kmeans_p"].predict(Xp)

    v_names = [models["value_map"][int(i)] for i in v_ids]
    p_names = [models["mix_map"][int(i)] for i in p_ids]

    segments = [FINAL_SEGMENT_LOOKUP[(v, m)] for v, m in zip(v_names, p_names)]
    folded = models.get("folded", {})
    segments = [folded.get(s, s) for s in segments]

    return pd.DataFrame({
        "value_cluster": v_ids,
        "mix_cluster": p_ids,
        "segment": segments,
    }, index=customer_df.index)
