"""
Batch scoring: validate an uploaded customer CSV and classify every row.

Kept separate from app.py so the logic is testable without a browser.
Expects the same schema as data/customer_info.csv (the raw export a
marketing team would have), runs the documented predict path:
preprocessing.build_features -> clustering.predict_segment.

Scoring uses the **frozen** preprocessing parameters saved at training time
(medians, gender mode, spend caps) rather than re-deriving them from the
uploaded file. That is what makes a customer's segment a property of the
customer instead of a property of the batch they arrived in: upload the same
row alone or inside 33,000 others and the answer is identical.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg         # noqa: E402
from src import preprocessing as pp   # noqa: E402
from src import clustering as clu     # noqa: E402

MAX_ROWS = 50_000

# Verified empirically: build_features fails without each of these (plus
# customer_id for the output join and lifetime_total_distinct_products,
# which predict_segment needs as a value-perspective feature). Anything
# else in the file is carried through untouched and ignored.
REQUIRED_COLUMNS = [
    "customer_id", "customer_name", "customer_gender", "customer_birthdate",
    "kids_home", "teens_home", "number_complaints",
    "distinct_stores_visited", "lifetime_spend_groceries",
    "lifetime_spend_electronics", "typical_hour",
    "lifetime_spend_vegetables", "lifetime_spend_nonalcohol_drinks",
    "lifetime_spend_alcohol_drinks", "lifetime_spend_meat",
    "lifetime_spend_fish", "lifetime_spend_hygiene",
    "lifetime_spend_videogames", "lifetime_spend_petfood",
    "lifetime_total_distinct_products",
    "percentage_of_products_bought_promotion",
    "year_first_transaction", "loyalty_card_number",
]


def score_batch(raw_bytes: bytes, models=None) -> tuple[pd.DataFrame | None, str | None]:
    """Parse, validate and score an uploaded CSV.

    Returns (result_df, None) on success or (None, error_message) on any
    problem the user can act on. Never raises for bad user input.
    """
    try:
        raw = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception:
        return None, ("Could not parse the file as CSV. Export a plain "
                      "comma-separated file and try again.")

    if raw.empty:
        return None, "The file parsed but contains no rows."
    if len(raw) > MAX_ROWS:
        return None, (f"The file has {len(raw):,} rows; the demo caps at "
                      f"{MAX_ROWS:,}. Split the file and score in parts.")

    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        return None, ("Missing required columns: " + ", ".join(missing)
                      + ". The expected schema matches customer_info.csv; "
                      "download the sample file for a template.")

    # Columns that must parse as numbers. Catching this here gives the user a
    # column name; letting it reach build_features surfaces a pandas dtype
    # message that says nothing about which column to look at.
    numeric = [c for c in REQUIRED_COLUMNS
               if c not in ("customer_id", "customer_name",
                            "customer_gender", "customer_birthdate")]
    non_numeric = [
        c for c in numeric
        if not pd.api.types.is_numeric_dtype(pd.to_numeric(raw[c], errors="coerce"))
        or pd.to_numeric(raw[c], errors="coerce").isna().sum() > raw[c].isna().sum()
    ]
    if non_numeric:
        return None, ("These columns contain values that are not numbers: "
                      + ", ".join(non_numeric)
                      + ". Blank cells are fine; text is not. Fix those cells "
                        "and re-upload.")

    if models is None:
        models = clu.load_models()

    params = models.get("preproc_params")
    if params is None:
        return None, ("The saved model is missing its preprocessing parameters. "
                      "Re-run notebook 03 (or python app/prep_app_data.py) to "
                      "regenerate outputs/models/.")

    try:
        feats = pp.build_features(raw, params)
    except Exception as exc:  # surface, don't crash the app
        return None, f"Could not build features from the file: {exc}"

    needed = cfg.VALUE_FEATURES + cfg.PRODUCT_MIX_FEATURES
    bad = feats[needed].isna().any()
    if bad.any():
        cols = ", ".join(bad[bad].index)
        return None, (f"Some rows have values that could not be resolved in: {cols}. "
                      "Check those columns for text or blank entries and re-upload.")

    try:
        pred = clu.predict_segment(feats, models=models)
    except Exception as exc:  # surface, don't crash the app
        return None, f"Scoring failed while processing the data: {exc}"

    result = pd.DataFrame({
        "customer_id": raw["customer_id"].values,
        "segment": pred["segment"].values,
        "value_cluster": pred["value_cluster"].values,
        "mix_cluster": pred["mix_cluster"].values,
    })
    return result, None
