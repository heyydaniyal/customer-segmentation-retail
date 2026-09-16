"""
Tests for src.preprocessing.

Run with:
    pytest tests/

The tests use small synthetic DataFrames so they don't depend on the
real customer_info.csv. The goal is to lock down the invariants that
the rest of the project relies on.
"""

import numpy as np
import pandas as pd
import pytest

from src import preprocessing as pp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_minimal_customer(**overrides):
    """Build a single-row customer DataFrame with required columns."""
    base = {
        "customer_id": 1,
        "customer_name": "Bsc. Some Name",
        "customer_gender": "female",
        "customer_birthdate": "01/15/1990 10:30 AM",
        "kids_home": 1.0,
        "teens_home": 0.0,
        "number_complaints": 0.0,
        "distinct_stores_visited": 3.0,
        "lifetime_spend_groceries": 1000.0,
        "lifetime_spend_electronics": 200.0,
        "lifetime_spend_vegetables": 100.0,
        "lifetime_spend_nonalcohol_drinks": 50.0,
        "lifetime_spend_alcohol_drinks": 50.0,
        "lifetime_spend_meat": 80.0,
        "lifetime_spend_fish": 40.0,
        "lifetime_spend_hygiene": 30.0,
        "lifetime_spend_videogames": 0.0,
        "lifetime_spend_petfood": 0.0,
        "lifetime_total_distinct_products": 50.0,
        "year_first_transaction": 2020.0,
        "loyalty_card_number": 1.0,
        "latitude": 38.75,
        "longitude": -9.15,
        "percentage_of_products_bought_promotion": 0.3,
        "typical_hour": 14.0,
    }
    base.update(overrides)
    return pd.DataFrame([base])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_extract_degree_recognises_known_prefixes():
    """Bsc., Msc., Phd. (case-insensitive) should map to their canonical forms;
    everything else should fall into the 'None' bucket."""
    df = pd.DataFrame({"customer_name": [
        "Bsc. Alice Brown",
        "Msc. Bob Carter",
        "Phd. Carol Dixon",
        "BSC. Dan Evans",         # lowercase variant
        "Mr. Eve Frost",          # not a known prefix
        "Just a Name",            # no prefix at all
        None,                     # missing name
    ]})
    out = pp.extract_degree(df)
    assert out["degree"].tolist() == [
        "Bsc", "Msc", "Phd", "Bsc", "None", "None", "None"
    ]


def test_share_columns_sum_to_one():
    """For any customer with non-zero spend, share_* columns must sum to 1."""
    df = make_minimal_customer()
    out = pp.build_features(df)
    share_cols = [c for c in out.columns if c.startswith("share_")]
    row_sum = out[share_cols].iloc[0].sum()
    # Allow for floating-point tolerance
    assert abs(row_sum - 1.0) < 1e-9, f"Share columns sum to {row_sum}, not 1.0"


def test_future_year_clipped_and_promo_clipped():
    """Impossible values in source data must be corrected by the pipeline."""
    df = make_minimal_customer(
        year_first_transaction=2029.0,
        percentage_of_products_bought_promotion=-0.5,
    )
    out = pp.build_features(df)
    # year_first_transaction is clipped to 2026, tenure must be >= 0
    assert out["year_first_transaction"].iloc[0] == 2026.0
    assert out["tenure_years"].iloc[0] == 0.0
    # Promo percentage clipped to [0, 1]
    assert out["promo_pct"].iloc[0] >= 0.0
    assert out["promo_pct"].iloc[0] <= 1.0


def test_loyalty_card_becomes_binary_flag():
    """loyalty_card_number value of 1.0 -> has_loyalty_card = 1;
    missing -> 0."""
    df_with = make_minimal_customer(loyalty_card_number=1.0)
    df_without = make_minimal_customer(loyalty_card_number=np.nan)
    out_with = pp.build_features(df_with)
    out_without = pp.build_features(df_without)
    assert out_with["has_loyalty_card"].iloc[0] == 1
    assert out_without["has_loyalty_card"].iloc[0] == 0


def test_missing_spend_imputed_as_zero():
    """Spend columns with missing values must become 0, not the median."""
    df = make_minimal_customer(lifetime_spend_videogames=np.nan)
    out = pp.build_features(df)
    assert out["lifetime_spend_videogames"].iloc[0] == 0.0


def test_typical_hour_cyclic_encoding_close_for_adjacent_hours():
    """Sin/cos encoding should put 23:00 and 00:00 closer than 23:00 and 12:00."""
    df_23 = make_minimal_customer(typical_hour=23.0)
    df_00 = make_minimal_customer(typical_hour=0.0)
    df_12 = make_minimal_customer(typical_hour=12.0)
    out_23 = pp.build_features(df_23).iloc[0]
    out_00 = pp.build_features(df_00).iloc[0]
    out_12 = pp.build_features(df_12).iloc[0]

    def dist(a, b):
        return ((a["typical_hour_sin"] - b["typical_hour_sin"]) ** 2 +
                (a["typical_hour_cos"] - b["typical_hour_cos"]) ** 2) ** 0.5

    assert dist(out_23, out_00) < dist(out_23, out_12), \
        "Cyclic encoding does not put 23:00 closer to 00:00 than to 12:00"


# ---------------------------------------------------------------------------
# Frozen preprocessing parameters
# ---------------------------------------------------------------------------
def test_fitted_params_make_scoring_batch_independent():
    """The whole point of freezing medians and caps: a customer's features must
    not depend on who else is in the file.

    Before the params were persisted, the 99th-percentile cap was recomputed
    from each uploaded batch, so a high spender clipped in the full population
    was left unclipped in a small file - and could land in a different segment.
    """
    big = pd.concat(
        [make_minimal_customer(customer_id=i, lifetime_spend_electronics=float(i))
         for i in range(1, 201)],
        ignore_index=True,
    )
    params = pp.fit_preprocessing_params(big)

    target = big[big["customer_id"] == 200]
    alone = pp.build_features(target, params)
    in_batch = pp.build_features(big, params)
    in_batch = in_batch[in_batch["customer_id"] == 200]

    for col in ["total_spend_log", "share_electronics", "promo_pct"]:
        assert abs(alone[col].iloc[0] - in_batch[col].iloc[0]) < 1e-12, \
            f"{col} changed with batch composition"


def test_fitted_params_beat_self_fitting_on_a_small_batch():
    """Self-fitting on a small batch really does give a different answer, which
    is why the frozen path exists. If this ever stops being true the frozen
    params are not doing anything."""
    big = pd.concat(
        [make_minimal_customer(customer_id=i, lifetime_spend_electronics=float(i * 10))
         for i in range(1, 201)],
        ignore_index=True,
    )
    params = pp.fit_preprocessing_params(big)
    target = big[big["customer_id"] == 200]

    frozen = pp.build_features(target, params)["lifetime_spend_electronics"].iloc[0]
    self_fitted = pp.build_features(target)["lifetime_spend_electronics"].iloc[0]

    assert frozen < self_fitted, \
        "the frozen cap should clip this high spender; a self-fitted one cannot"
    assert abs(frozen - params["caps"]["lifetime_spend_electronics"]) < 1e-9


def test_median_imputation_uses_frozen_value_not_batch_median():
    """A single row with a missing value must still impute - previously this
    produced NaN (median of nothing) and a raw sklearn traceback."""
    fit_on = pd.concat(
        [make_minimal_customer(customer_id=i, distinct_stores_visited=float(i % 5 + 1))
         for i in range(1, 101)],
        ignore_index=True,
    )
    params = pp.fit_preprocessing_params(fit_on)

    one = make_minimal_customer(distinct_stores_visited=np.nan)
    out = pp.build_features(one, params)
    assert out["distinct_stores_visited"].notna().all()
    assert out["distinct_stores_visited"].iloc[0] == params["medians"]["distinct_stores_visited"]


def test_drop_identifiers_removes_pii_but_keeps_the_join_key():
    """Nothing persisted to outputs/ should carry a name, birthdate or location."""
    out = pp.build_features(make_minimal_customer())
    slim = pp.drop_identifiers(out)

    for col in ["customer_name", "customer_birthdate", "latitude", "longitude",
                "loyalty_card_number"]:
        assert col not in slim.columns, f"{col} survived drop_identifiers"

    assert "customer_id" in slim.columns, "the basket join needs customer_id"
    assert "age" in slim.columns, "age is derived from birthdate and must survive"
    assert "degree" in slim.columns, "degree is derived from name and must survive"
    assert "has_loyalty_card" in slim.columns


def test_incomplete_params_raise_instead_of_silently_self_fitting():
    """A half-populated params dict must fail loudly.

    Falling back to fitting the missing statistic from whatever rows are on
    hand would quietly reintroduce the batch-dependence these params exist to
    remove - the worst outcome, because nothing signals it.
    """
    full = pp.fit_preprocessing_params(
        pd.concat([make_minimal_customer(customer_id=i) for i in range(1, 51)],
                  ignore_index=True)
    )
    one = make_minimal_customer()

    with pytest.raises(ValueError, match="incomplete"):
        pp.build_features(one, {"caps": full["caps"], "gender_mode": "female"})
    with pytest.raises(ValueError, match="incomplete"):
        pp.build_features(one, {"medians": full["medians"], "gender_mode": "female"})
    with pytest.raises(ValueError, match="medians for"):
        pp.build_features(one, {**full, "medians": {
            k: v for k, v in full["medians"].items() if k != "age"}})
    with pytest.raises(ValueError, match="spend caps for"):
        pp.build_features(one, {**full, "caps": {
            k: v for k, v in full["caps"].items() if k != "lifetime_spend_fish"}})
    with pytest.raises(TypeError):
        pp.build_features(one, "not a dict")

    # the complete dict still works
    assert len(pp.build_features(one, full)) == 1
