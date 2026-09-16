"""
Tests for src.clustering.

Run with:
    pytest tests/

These lock down the part of the pipeline most likely to break silently:
the content-based cluster naming and the final segment assignment. We build
the real features once (the dataset is small) and assert on the naming and
coverage rather than on exact cluster membership, which can vary slightly
with K-Means seeding.
"""

import warnings

import pytest

from src import preprocessing as pp
from src import clustering as clu
from src import config as cfg

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def clustered_df():
    """Build features and fit the two real perspectives once for all tests."""
    df = pp.build_features(pp.load_customer_info())
    _, Xv = clu.scale_perspective(df, cfg.VALUE_FEATURES)
    _, lv = clu.fit_kmeans(Xv, n_clusters=3)
    df["value_cluster"] = lv
    _, Xp = clu.scale_perspective(df, cfg.PRODUCT_MIX_FEATURES)
    _, lp = clu.fit_kmeans(Xp, n_clusters=4)
    df["mix_cluster"] = lp
    return df


def test_value_naming_returns_expected_three_names(clustered_df):
    """name_value_clusters must return exactly the three engagement names,
    one per cluster, regardless of how K-Means numbered them."""
    mapping = clu.name_value_clusters(clustered_df)
    assert len(mapping) == 3
    assert set(mapping.values()) == {"engaged", "promo_hunter", "routine"}


def test_mix_naming_returns_expected_four_names(clustered_df):
    """name_mix_clusters must return exactly the four product-mix names."""
    mapping = clu.name_mix_clusters(clustered_df)
    assert len(mapping) == 4
    assert set(mapping.values()) == {
        "tech", "family_provisioner", "young_parents", "grocery_generalist"
    }


def test_naming_raises_on_wrong_cluster_count(clustered_df):
    """If the number of clusters does not match what the namer expects, it must
    fail loudly rather than silently mislabel.

    This is a ValueError, not an assert: `python -O` strips asserts, and under
    -O the old assert-based guard silently returned three names for a
    four-cluster input, dropping a whole cluster."""
    _, Xv = clu.scale_perspective(clustered_df, cfg.VALUE_FEATURES)
    _, wrong = clu.fit_kmeans(Xv, n_clusters=4)
    tmp = clustered_df.copy()
    tmp["value_cluster"] = wrong
    with pytest.raises(ValueError):
        clu.name_value_clusters(tmp)


def test_assign_final_segments_covers_every_row(clustered_df):
    """Every customer must receive a non-null final segment, and all final
    segments must be at least MIN_SEGMENT_SIZE after folding."""
    segments, naming = clu.assign_final_segments(clustered_df)
    assert len(segments) == len(clustered_df)
    assert segments.notna().all()
    sizes = segments.value_counts()
    assert sizes.min() >= clu.MIN_SEGMENT_SIZE, (
        f"segment {sizes.idxmin()} has only {sizes.min()} customers, "
        f"below MIN_SEGMENT_SIZE={clu.MIN_SEGMENT_SIZE}"
    )


def test_no_high_value_label_leaks(clustered_df):
    """The value axis measures engagement, not spend; the misleading
    'High-Value' label must not appear in any final segment name."""
    segments, _ = clu.assign_final_segments(clustered_df)
    names = " ".join(segments.unique())
    assert "High-Value" not in names
    assert "Healthy" not in names  # renamed to 'Young Parents'


def test_cluster_profile_uses_ratio_scale_for_normal_features(clustered_df):
    """Regression test for the mask-alignment bug.

    `near_zero` is indexed by feature name while the intermediate frames are
    indexed by cluster id, and DataFrame.where aligns a Series condition on the
    index (ignoring axis=), so passing it directly produced an all-NaN
    condition and forced every feature onto the standardised-difference scale.
    The heatmap still said "% of overall mean", and the product-mix panel
    showed negative percentages of a mean.

    None of the value features are near-zero-mean, so every row must be a plain
    ratio of the overall mean.
    """
    profile = clu.cluster_profile(
        clustered_df, clustered_df["value_cluster"].values, cfg.VALUE_FEATURES
    )
    expected = (clustered_df.groupby("value_cluster")[cfg.VALUE_FEATURES].mean()
                / clustered_df[cfg.VALUE_FEATURES].mean() * 100).round(1).T

    assert profile.attrs["zscored_features"] == [], \
        "no value feature is near-zero-mean, so none should be z-scored"
    assert (profile.to_numpy() > 0).all(), \
        "a ratio of a positive mean can never be negative"
    for i, feature in enumerate(cfg.VALUE_FEATURES):
        for j in range(profile.shape[1]):
            assert abs(profile.iloc[i, j] - expected.iloc[i, j]) < 0.11, \
                f"{feature} is not on the ratio-of-mean scale"


def test_cluster_profile_zscores_only_near_zero_mean_features(clustered_df):
    """The cyclic hour encoding averages near zero, so a ratio-of-mean would
    explode. That feature - and only that feature - should switch scale, and
    the switch must be recorded so the plot can label it."""
    profile = clu.cluster_profile(
        clustered_df, clustered_df["mix_cluster"].values, cfg.DEMOGRAPHIC_FEATURES
    )
    zscored = profile.attrs["zscored_features"]
    assert "typical_hour_sin" in zscored
    assert "age" not in zscored and "kids_home" not in zscored
