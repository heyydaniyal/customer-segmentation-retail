"""
Baseline segmentations.

We compare our K-Means clustering against a simple RFM-style heuristic.
If the heuristic produced the same segments as K-Means, the ML wouldn't
be adding value. By computing the Adjusted Rand Index between the two
segmentations we can quantify how much extra information the clustering
captures.

RFM stands for Recency, Frequency, Monetary, the standard customer
segmentation heuristic in retail analytics. Our data does not include
recency directly, so we use tenure (proxy for engagement length) and
substitute distinct_stores_visited for frequency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


def rfm_segments(df, r_col="tenure_years", f_col="distinct_stores_visited",
                 m_col="total_spend", n_bins=3):
    """
    Bucket each customer into an RFM segment using quantile binning.

    Returns:
        DataFrame with columns r_bin, f_bin, m_bin (each 1..n_bins) and
        a combined 'rfm_segment' string like '3-2-3'.
    """
    out = pd.DataFrame(index=df.index)
    # Higher tenure = better (more loyal)
    out["r_bin"] = pd.qcut(df[r_col].rank(method="first"), n_bins,
                            labels=range(1, n_bins + 1)).astype(int)
    out["f_bin"] = pd.qcut(df[f_col].rank(method="first"), n_bins,
                            labels=range(1, n_bins + 1)).astype(int)
    out["m_bin"] = pd.qcut(df[m_col].rank(method="first"), n_bins,
                            labels=range(1, n_bins + 1)).astype(int)
    out["rfm_segment"] = out["r_bin"].astype(str) + "-" + \
                         out["f_bin"].astype(str) + "-" + \
                         out["m_bin"].astype(str)
    return out


def coarse_rfm_label(rfm_df):
    """
    Collapse the 27 RFM combinations into a small set of named groups
    that roughly mirror our K-Means segments. Used to compare like with like.

    All three bins participate in the rule. An earlier version unpacked the
    recency bin and then never used it, which made this an FM heuristic
    wearing an RFM name — the baseline was weaker than it claimed to be.

    Returns a pd.Series of named labels.
    """
    def _label(row):
        r, f, m = row["r_bin"], row["f_bin"], row["m_bin"]
        if m >= 3 and f >= 2 and r >= 2:
            return "rfm_high_value"
        if m == 1 and f == 1 and r == 1:
            return "rfm_lapsed_low"
        if r >= 3 and f >= 3:
            return "rfm_loyal_frequent"
        if f >= 3:
            return "rfm_frequent"
        return "rfm_middle"
    return rfm_df.apply(_label, axis=1)


def compare_segmentations(labels_a, labels_b):
    """Adjusted Rand Index between two label vectors."""
    return adjusted_rand_score(labels_a, labels_b)


# ---------------------------------------------------------------------------
# Compositional-data transformation (CLR) for the product-mix perspective
# ---------------------------------------------------------------------------
def clr_transform(share_df, eps=1e-6):
    """
    Centred log-ratio transformation for compositional data.

    The product-mix features (share columns) lie on the simplex: each row
    sums to 1 and each value is in [0, 1]. Euclidean distance on the
    simplex is not the natural metric for compositional data. The CLR
    transformation maps the simplex to Euclidean space by:

        clr(x_i) = log(x_i / geometric_mean(x))

    After CLR, standard K-Means with Euclidean distance is geometrically
    appropriate. We add a small `eps` to avoid log(0) for customers who
    spend zero in a category.

    Returns a DataFrame with the same shape as share_df.
    """
    arr = share_df.values + eps
    # Geometric mean per row
    log_arr = np.log(arr)
    gmean_log = log_arr.mean(axis=1, keepdims=True)
    clr = log_arr - gmean_log
    return pd.DataFrame(clr, index=share_df.index,
                        columns=[f"clr_{c}" for c in share_df.columns])
