"""
Preprocessing pipeline for customer_info.

The pipeline is split into small, testable functions. The main entry point
is build_features(), which composes everything and returns a clean DataFrame
ready for clustering.

Design notes
------------
- Each function takes a DataFrame and returns a copy. We avoid in-place
  mutation so notebooks can re-run cells in any order without surprises.
- Column-level decisions (which imputation, which outlier strategy) live
  here, not in config.py. config.py is for groupings; this file is for logic.
- Anything that would change the shape of the data (row drops, joins) is
  isolated into clearly named functions so it is obvious in the notebook
  what is happening.

Fitted vs. deterministic steps
-----------------------------
Two of these steps depend on statistics of the data they see: median
imputation and the 99th-percentile spend caps. Those statistics belong to the
*training* population and must be frozen, exactly like a scaler. If they were
recomputed at scoring time, the same customer could land in a different
segment depending on which file they happened to be uploaded alongside.

So the pipeline is split:

    params = fit_preprocessing_params(raw)   # once, on the training data
    df     = build_features(raw, params)     # anywhere, always the same answer

`build_features(raw)` with no params still fits-then-transforms, which is the
right behaviour for the training run and keeps the notebooks readable.
`clustering.save_models` persists the params next to the scalers, and
`predict_segment`'s callers load them back, so inference never re-derives a
training statistic.
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd

from . import config as cfg


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_customer_info(path=None) -> pd.DataFrame:
    """Load the raw customer_info.csv file."""
    path = path or cfg.RAW_CUSTOMER_INFO
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Parsing and basic cleaning
# ---------------------------------------------------------------------------
def parse_birthdate(df: pd.DataFrame, ref_date: str = cfg.REFERENCE_DATE) -> pd.DataFrame:
    """Convert birthdate string into a numeric age in years."""
    out = df.copy()
    parsed = pd.to_datetime(
        out["customer_birthdate"],
        format="%m/%d/%Y %I:%M %p",
        errors="coerce",
    )
    out["age"] = (pd.Timestamp(ref_date) - parsed).dt.days / 365.25
    return out


def extract_degree(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the academic degree from customer_name.

    Names look like 'Bsc. Crystal Kitchens' or 'John Kelling'. We grab the
    leading token if it ends in '.' and is one of the known degree prefixes.
    Missing prefix is mapped to the 'None' category, which is itself
    informative.
    """
    out = df.copy()
    pattern = re.compile(r"^(Bsc|Msc|Phd)\.\s", flags=re.IGNORECASE)

    def _degree(name):
        if not isinstance(name, str):
            return "None"
        match = pattern.match(name)
        if match is None:
            return "None"
        return match.group(1).capitalize()

    out["degree"] = out["customer_name"].apply(_degree)
    return out


def transform_loyalty_card(df: pd.DataFrame) -> pd.DataFrame:
    """
    The loyalty_card_number column only ever holds the value 1.0 or NaN, so
    as a numeric feature it carries no information. The presence/absence
    pattern is the actual signal.
    """
    out = df.copy()
    out["has_loyalty_card"] = out["loyalty_card_number"].notna().astype(int)
    return out


def clip_first_transaction_year(
    df: pd.DataFrame,
    max_year: int = cfg.MAX_VALID_FIRST_TRANSACTION_YEAR,
) -> pd.DataFrame:
    """
    Around 3 percent of rows have first_transaction_year in the future
    (2027 to 2029). These are data errors. We clip to the maximum valid year
    rather than dropping, because the remaining columns for those customers
    look fine.
    """
    out = df.copy()
    out["year_first_transaction"] = out["year_first_transaction"].clip(upper=max_year)
    out["tenure_years"] = max_year - out["year_first_transaction"]
    return out


def clip_promo_percentage(df: pd.DataFrame) -> pd.DataFrame:
    """
    percentage_of_products_bought_promotion has values as low as -1.27, which
    is impossible. Clip to the valid [0, 1] range.
    """
    out = df.copy()
    out["percentage_of_products_bought_promotion"] = out[
        "percentage_of_products_bought_promotion"
    ].clip(lower=0, upper=1)
    return out


# ---------------------------------------------------------------------------
# Missing value imputation
# ---------------------------------------------------------------------------
def impute_missing(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """
    Column-by-column imputation. The strategy for each column is chosen
    based on the meaning of the column, not a one-size-fits-all default.

    - Spend columns: missing implies zero spend in that category.
    - Counts of kids, teens, complaints: missing implies zero.
    - Continuous numerics (stores visited, typical hour, promo %): median.
    - Age: median, since missingness is well under 1 percent.
    - Gender: mode.

    The zero-fills are deterministic. The medians and the gender mode are
    *fitted* statistics: pass `params` (from fit_preprocessing_params) to use
    the frozen training values. With params=None the values are computed from
    `df` itself, which is correct only when `df` is the training population.
    """
    out = df.copy()

    for col in cfg.SPEND_FILL_ZERO:
        out[col] = out[col].fillna(0)

    for col in cfg.COUNT_FILL_ZERO:
        out[col] = out[col].fillna(0)

    medians = (params or {}).get("medians")
    gender_mode = (params or {}).get("gender_mode")

    for col in cfg.MEDIAN_IMPUTE:
        fill = medians[col] if medians else out[col].median()
        out[col] = out[col].fillna(fill)

    if out["age"].isna().any():
        fill = medians["age"] if medians else out["age"].median()
        out["age"] = out["age"].fillna(fill)

    if out["customer_gender"].isna().any():
        fill = gender_mode or out["customer_gender"].mode().iloc[0]
        out["customer_gender"] = out["customer_gender"].fillna(fill)

    return out


# ---------------------------------------------------------------------------
# Outlier handling
# ---------------------------------------------------------------------------
def validate_params(params: dict) -> dict:
    """
    Check a preprocessing-params dict is complete before it is used.

    Without this, a partially-populated dict (say one carrying `caps` but not
    `medians`) would silently fall back to fitting the missing statistic from
    whatever rows it was handed — quietly reintroducing exactly the
    batch-dependence these params exist to eliminate, with no error to notice.
    A half-frozen pipeline is worse than an obviously unfrozen one, so this
    raises instead.
    """
    if not isinstance(params, dict):
        raise TypeError(f"preprocessing params must be a dict, got {type(params).__name__}")

    missing = [k for k in ("medians", "gender_mode", "caps") if not params.get(k)]
    if missing:
        raise ValueError(
            f"preprocessing params are incomplete, missing: {missing}. "
            "Regenerate them with fit_preprocessing_params(), or pass params=None "
            "to fit-and-transform deliberately."
        )

    missing_medians = [c for c in cfg.MEDIAN_IMPUTE + ["age"]
                       if c not in params["medians"]]
    if missing_medians:
        raise ValueError(f"preprocessing params missing medians for: {missing_medians}")

    missing_caps = [c for c in capped_spend_columns() if c not in params["caps"]]
    if missing_caps:
        raise ValueError(f"preprocessing params missing spend caps for: {missing_caps}")

    return params


def capped_spend_columns() -> list[str]:
    """The nine spend columns that get a 99th-percentile cap (all but groceries)."""
    return [c for c in cfg.SPEND_COLUMNS if c != "lifetime_spend_groceries"]


def cap_spend_outliers(
    df: pd.DataFrame,
    params: dict | None = None,
    columns: list[str] | None = None,
    quantile: float = 0.99,
) -> pd.DataFrame:
    """
    Cap extreme spend values at a high quantile.

    The cap values are a *fitted* statistic. Pass `params` (from
    fit_preprocessing_params) to clip at the frozen training thresholds. With
    params=None the quantiles are taken from `df` itself, which is correct
    only when `df` is the training population — on a small uploaded batch a
    self-fitted 99th percentile is close to that batch's maximum and clips
    almost nothing, so the same customer would score differently depending on
    who they were uploaded with.

    The spend distributions are heavily right-skewed. A handful of customers
    with very high spend would otherwise pull K-Means centroids around. We
    cap each of the non-grocery spend columns at its own 99th percentile.
    Because we cap nine columns independently, the union of affected rows is
    larger than 1 percent of customers, though each individual column loses
    only its top ~1 percent of values.

    We do NOT cap lifetime_spend_groceries here. Groceries is by far the
    largest component of total_spend (about 69 percent), and total_spend is
    log-transformed downstream (total_spend_log in add_engineered_features),
    which already tames the grocery-driven skew without a hard cap. Capping
    the other columns is still useful because they feed the product-mix
    shares and the raw value features, where extreme tails would distort the
    scaler.
    """
    out = df.copy()
    caps = (params or {}).get("caps")
    columns = columns or (list(caps) if caps else capped_spend_columns())

    for col in columns:
        cap = caps[col] if caps else out[col].quantile(quantile)
        out[col] = out[col].clip(upper=cap)

    return out


def cap_family_counts(df: pd.DataFrame, cap: int = 4) -> pd.DataFrame:
    """
    kids_home and teens_home have tails up to 8 and 6. We cap at 4 to keep
    the 'large family' signal without letting a handful of customers with
    extreme counts dominate distance calculations.
    """
    out = df.copy()
    out["kids_home"] = out["kids_home"].clip(upper=cap)
    out["teens_home"] = out["teens_home"].clip(upper=cap)
    return out


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derived features used by the clustering perspectives.

    - total_spend / total_spend_log: overall value
    - share_*: each category as a fraction of total spend (product mix)
    - typical_hour_sin / typical_hour_cos: cyclic encoding for hour of day
    - degree dummies, is_female: lightweight one-hot
    - promo_pct: shorter alias to keep code readable
    """
    out = df.copy()

    out["total_spend"] = out[cfg.SPEND_COLUMNS].sum(axis=1)
    # log1p safely handles total_spend == 0 (rare/none in this dataset).
    out["total_spend_log"] = np.log1p(out["total_spend"])

    # Per-category share. For customers with total_spend == 0 (theoretically
    # possible, none observed here), the natural share is undefined. Rather
    # than fill with all-zeros (which would put them at a distorted point in
    # share-space far from anyone else), we fill with the uniform share
    # 1/N_CATEGORIES so they sit at the centre of the simplex.
    n_categories = len(cfg.SPEND_COLUMNS)
    zero_mask = out["total_spend"] == 0
    safe_total = out["total_spend"].replace(0, np.nan)
    for col in cfg.SPEND_COLUMNS:
        short = col.replace("lifetime_spend_", "")
        share = (out[col] / safe_total).fillna(1.0 / n_categories)
        out[f"share_{short}"] = share
    if zero_mask.any():
        print(f"  Note: {zero_mask.sum()} customer(s) with zero total spend "
              f"received uniform shares (1/{n_categories} each).")

    # Cyclic encoding so 23:00 and 00:00 are close in feature space.
    out["typical_hour_sin"] = np.sin(2 * np.pi * out["typical_hour"] / 24)
    out["typical_hour_cos"] = np.cos(2 * np.pi * out["typical_hour"] / 24)

    # Degree dummies. Including 'None' as its own column on purpose so all
    # four categories are represented symmetrically.
    for d in ["Bsc", "Msc", "Phd", "None"]:
        out[f"degree_{d}"] = (out["degree"] == d).astype(int)

    out["is_female"] = (out["customer_gender"] == "female").astype(int)

    out["promo_pct"] = out["percentage_of_products_bought_promotion"]

    return out


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def _clean_deterministic(raw: pd.DataFrame) -> pd.DataFrame:
    """
    The part of the pipeline that depends only on each row's own values:
    parsing, extraction, and clipping to physically valid ranges. No fitted
    statistics, so this is identical whether it runs on 1 row or 33,038.
    """
    df = parse_birthdate(raw)
    df = extract_degree(df)
    df = transform_loyalty_card(df)
    df = clip_first_transaction_year(df)
    df = clip_promo_percentage(df)
    return df


def fit_preprocessing_params(raw: pd.DataFrame, quantile: float = 0.99) -> dict:
    """
    Learn the training-time statistics the pipeline needs: the medians used
    for imputation, the gender mode, and the per-column spend caps.

    Caps are measured *after* imputation because that is the order the
    pipeline applies them in, and the spend zero-fills shift the quantiles.

    Returns a plain dict of JSON-serialisable values so it can be persisted
    alongside the scalers and reloaded at inference time.
    """
    return _fit_from_clean(_clean_deterministic(raw), len(raw), quantile)


def _fit_from_clean(df: pd.DataFrame, raw_rows: int, quantile: float = 0.99) -> dict:
    """Fit the statistics from an already-cleaned frame, so the training path
    does not run the deterministic cleaning twice."""
    medians = {col: float(df[col].median()) for col in cfg.MEDIAN_IMPUTE}
    medians["age"] = float(df["age"].median())
    gender_mode = str(df["customer_gender"].mode().iloc[0])

    imputed = impute_missing(
        df, {"medians": medians, "gender_mode": gender_mode}
    )
    caps = {
        col: float(imputed[col].quantile(quantile))
        for col in capped_spend_columns()
    }

    return {
        "quantile": quantile,
        "medians": medians,
        "gender_mode": gender_mode,
        "caps": caps,
        "n_fit_rows": int(raw_rows),
    }


def build_features(raw: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """
    Run the full preprocessing pipeline.

    Order matters: parsing first, then cleaning fixes the columns we will
    impute against, then imputation, then outlier handling, finally
    feature engineering which depends on clean inputs.

    Pass `params` from fit_preprocessing_params to transform with frozen
    training statistics. Leave it None to fit-and-transform, which is what
    the training run in notebook 02/03 wants.
    """
    df = _clean_deterministic(raw)
    if params is None:
        params = _fit_from_clean(df, raw_rows=len(raw))
    else:
        validate_params(params)
    df = impute_missing(df, params)
    df = cap_spend_outliers(df, params)
    df = cap_family_counts(df)
    df = add_engineered_features(df)
    return df


def build_features_with_params(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Training-run convenience: return both the features and the fitted params.

    Cleans once and fits from the cleaned frame, rather than calling
    fit_preprocessing_params and build_features separately (which would run the
    deterministic cleaning twice over the whole population).
    """
    clean = _clean_deterministic(raw)
    params = _fit_from_clean(clean, raw_rows=len(raw))
    df = impute_missing(clean, params)
    df = cap_spend_outliers(df, params)
    df = cap_family_counts(df)
    df = add_engineered_features(df)
    return df, params


def drop_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove per-person identifiers and superseded raw columns before anything
    is written to disk.

    Name, birthdate and coordinates are needed to *derive* features (age,
    degree) but nothing downstream reads them, so no persisted artefact
    should carry them. `customer_id` stays because the basket join needs it.
    """
    drop = [
        c for c in cfg.IDENTIFIER_COLUMNS + cfg.SUPERSEDED_COLUMNS
        if c in df.columns
    ]
    return df.drop(columns=drop)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a small DataFrame summarising missingness per column."""
    miss = df.isnull().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    return pd.DataFrame(
        {
            "missing": miss,
            "pct_missing": (miss / len(df) * 100).round(2),
        }
    )
