"""
Central configuration for the customer segmentation project.

Keeping feature groupings, paths, and constants in one place so notebooks
stay in sync. If we decide a feature belongs in a
different perspective, we change it here and everything downstream follows.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"

RAW_CUSTOMER_INFO = DATA_DIR / "customer_info.csv"
RAW_CUSTOMER_BASKET = DATA_DIR / "customer_basket.csv"

PROCESSED_CUSTOMER_INFO = OUTPUT_DIR / "customer_info_processed.parquet"
FINAL_SEGMENTS_CSV = OUTPUT_DIR / "customer_segments.csv"

# ---------------------------------------------------------------------------
# Reference date for age and tenure calculations
# ---------------------------------------------------------------------------
# We anchor everything to a fixed reference date so re-runs are reproducible.
# The dataset was last updated in 2026, so we use mid-2026.
REFERENCE_DATE = "2026-05-19"
MAX_VALID_FIRST_TRANSACTION_YEAR = 2026

# ---------------------------------------------------------------------------
# Columns that must never reach a persisted artefact
# ---------------------------------------------------------------------------
# The raw export carries per-person identifiers. They are needed to derive
# features (age from birthdate, degree from name) but nothing downstream reads
# them, so preprocessing.drop_identifiers() strips them before anything is
# written to outputs/. customer_id survives because the basket join needs it.
IDENTIFIER_COLUMNS = [
    "customer_name",
    "customer_birthdate",
    "latitude",
    "longitude",
]

# Raw columns fully superseded by an engineered replacement.
SUPERSEDED_COLUMNS = [
    "loyalty_card_number",   # -> has_loyalty_card
]

# ---------------------------------------------------------------------------
# Raw column groupings
# ---------------------------------------------------------------------------
SPEND_COLUMNS = [
    "lifetime_spend_groceries",
    "lifetime_spend_electronics",
    "lifetime_spend_vegetables",
    "lifetime_spend_nonalcohol_drinks",
    "lifetime_spend_alcohol_drinks",
    "lifetime_spend_meat",
    "lifetime_spend_fish",
    "lifetime_spend_hygiene",
    "lifetime_spend_videogames",
    "lifetime_spend_petfood",
]

# Columns where missing means "zero spend" rather than "unknown".
SPEND_FILL_ZERO = SPEND_COLUMNS

# Columns where missing is most reasonably treated as zero count.
COUNT_FILL_ZERO = ["kids_home", "teens_home", "number_complaints"]

# Numeric columns where missing is imputed with the median.
MEDIAN_IMPUTE = [
    "distinct_stores_visited",
    "typical_hour",
    "percentage_of_products_bought_promotion",
]

# ---------------------------------------------------------------------------
# Clustering perspectives
# ---------------------------------------------------------------------------
# Each perspective is clustered independently. We then combine the perspective
# labels into a final segment label using a rules-based mapping (see
# clustering.py). Splitting features this way produces clusters that can each
# be described in plain language, which is the goal for the report.

# Perspective 1 - Value / engagement intensity
VALUE_FEATURES = [
    "total_spend_log",
    "lifetime_total_distinct_products",
    "distinct_stores_visited",
    "tenure_years",
    "promo_pct",
]

# Perspective 2 - Product mix (relative share of total spend per category)
PRODUCT_MIX_FEATURES = [
    "share_groceries",
    "share_electronics",
    "share_vegetables",
    "share_nonalcohol_drinks",
    "share_alcohol_drinks",
    "share_meat",
    "share_fish",
    "share_hygiene",
    "share_videogames",
    "share_petfood",
]

# Perspective 3 - Demographics / lifestyle
DEMOGRAPHIC_FEATURES = [
    "age",
    "kids_home",
    "teens_home",
    "number_complaints",
    "typical_hour_sin",
    "typical_hour_cos",
    "has_loyalty_card",
    # Degree is one-hot encoded rather than ordinal. Although Bsc < Msc < Phd
    # has a natural order, encoding it ordinally would impose an assumption
    # that the spending/behaviour gap between consecutive degrees is equal and
    # monotonic, which we have no evidence for. One-hot lets K-Means treat each
    # degree as its own direction; the cost is three extra columns, which is
    # cheap at this scale.
    "degree_Bsc",
    "degree_Msc",
    "degree_Phd",
    "degree_None",
    "is_female",
]

# ---------------------------------------------------------------------------
# Plotting defaults
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
FIGSIZE_DEFAULT = (9, 5)
FIGSIZE_WIDE = (12, 5)
