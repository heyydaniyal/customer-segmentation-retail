"""
Basket analysis with Apriori, run per customer segment.

We join the per-customer cluster labels onto customer_basket, then run
Apriori separately on the baskets of each segment. The resulting rules
become the basis for the promotional ideas in the report.

Why per-segment rather than global: a global rule like
'customers who buy chicken often buy rice' is not actionable for a marketing
campaign. A segment-specific rule like 'in the Routine Tech segment, buying
bluetooth headphones predicts buying airpods' lets us write a targeted
promotion for exactly that group.

A note on reading segment baskets
---------------------------------
Raw product frequency is a trap. Asparagus and airpods are the two most
common products overall (12.8% and 12.1% of all 100k baskets), so they sit
near the top of almost every segment's frequency list by construction, and
"segment X's top item is airpods" carries essentially no information.
`product_lift_by_segment` divides by the population rate, which is what
actually identifies a segment's character.
"""

from __future__ import annotations

import ast
from collections import Counter

import pandas as pd
from scipy.stats import spearmanr

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

from . import config as cfg


# ---------------------------------------------------------------------------
# Loading and parsing
# ---------------------------------------------------------------------------
def load_baskets(path=None) -> pd.DataFrame:
    """Load customer_basket.csv and parse list_of_goods into actual lists."""
    path = path or cfg.RAW_CUSTOMER_BASKET
    df = pd.read_csv(path)
    df["items"] = df["list_of_goods"].apply(ast.literal_eval)
    return df


def attach_segments(baskets: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'segment' column to the baskets DataFrame.

    segments is expected to have customer_id and a single label column. We
    inner-join because baskets without segment information are not useful
    for per-segment rule mining.
    """
    return baskets.merge(segments, on="customer_id", how="inner")


# ---------------------------------------------------------------------------
# Apriori per segment
# ---------------------------------------------------------------------------
def _encode_baskets(item_lists: pd.Series) -> pd.DataFrame:
    """Turn a series of item-lists into a one-hot DataFrame."""
    te = TransactionEncoder()
    array = te.fit_transform(item_lists.tolist())
    return pd.DataFrame(array, columns=te.columns_)


def rules_for_segment(
    item_lists: pd.Series,
    min_support: float = 0.02,
    min_lift: float = 1.2,
    metric: str = "lift",
    max_len: int = 3,
    min_occurrences: int = 30,
) -> pd.DataFrame:
    """
    Run Apriori on a single segment and return association rules.

    Parameters
    ----------
    item_lists : the basket lists for a single segment
    min_support : minimum fraction of baskets containing the itemset
    min_lift : minimum lift to keep a rule
    metric : metric passed to mlxtend.association_rules
    max_len : maximum itemset size, kept small to avoid combinatorial blowup
    min_occurrences : absolute floor on how many baskets must contain the
        itemset. For small segments min_support * n_baskets can be too few
        rows to support reliable lift estimates, so we raise the effective
        support until min_occurrences baskets are required. This protects
        against spurious "winning" rules in small segments.
    """
    n_baskets = len(item_lists)
    if n_baskets < 100:
        return pd.DataFrame()

    # Adaptive support: never accept fewer than min_occurrences baskets
    effective_support = max(min_support, min_occurrences / n_baskets)

    one_hot = _encode_baskets(item_lists)
    freq = apriori(one_hot, min_support=effective_support, use_colnames=True,
                   max_len=max_len)

    if freq.empty:
        return pd.DataFrame()

    rules = association_rules(freq, metric=metric, min_threshold=min_lift)
    if rules.empty:
        return rules

    rules = rules.sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)
    rules["antecedents"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(s)))
    rules["consequents"] = rules["consequents"].apply(lambda s: ", ".join(sorted(s)))
    # Add an absolute count column so users can see how many baskets support the rule
    rules["n_baskets"] = (rules["support"] * n_baskets).round().astype(int)
    return rules


def rules_by_segment(
    baskets_with_segments: pd.DataFrame,
    segment_col: str = "segment",
    min_support: float = 0.02,
    min_lift: float = 1.2,
    max_len: int = 3,
    min_occurrences: int = 30,
) -> dict[str, pd.DataFrame]:
    """
    Run rules_for_segment for each unique segment value.

    Returns a dict mapping segment name to its rules table.
    """
    results = {}
    for seg, group in baskets_with_segments.groupby(segment_col):
        results[seg] = rules_for_segment(
            group["items"],
            min_support=min_support,
            min_lift=min_lift,
            max_len=max_len,
            min_occurrences=min_occurrences,
        )
    return results


# ---------------------------------------------------------------------------
# Quick segment-level summaries useful for the report
# ---------------------------------------------------------------------------
def top_products_per_segment(
    baskets_with_segments: pd.DataFrame,
    segment_col: str = "segment",
    n: int = 15,
) -> pd.DataFrame:
    """Top N most-frequent products per segment, with frequency counts."""
    rows = []
    for seg, group in baskets_with_segments.groupby(segment_col):
        flat = [item for lst in group["items"] for item in lst]
        counts = pd.Series(flat).value_counts().head(n)
        for product, count in counts.items():
            rows.append(
                {
                    "segment": seg,
                    "product": product,
                    "count": int(count),
                    "share_of_baskets": round(count / len(group), 3),
                }
            )
    return pd.DataFrame(rows)


def product_lift_by_segment(
    baskets_with_segments: pd.DataFrame,
    segment_col: str = "segment",
    min_segment_share: float = 0.03,
    n: int = 8,
) -> pd.DataFrame:
    """
    Top N products per segment ranked by **lift against the population rate**,
    not by raw frequency.

    lift = (share of this segment's baskets containing the product)
           / (share of all baskets containing the product)

    A lift of 1.0 means the segment buys the product at exactly the rate
    everyone does. This is the correct way to read a segment's basket
    character: raw frequency mostly recovers the global bestseller list.

    `min_segment_share` requires a product to appear in at least that fraction
    of the segment's baskets before it can be ranked, which keeps rare
    products with noisy ratios out of the top of the table.
    """
    counts = Counter(item for lst in baskets_with_segments["items"] for item in lst)
    n_total = len(baskets_with_segments)
    population = pd.Series({p: c / n_total for p, c in counts.items()})

    rows = []
    for seg, group in baskets_with_segments.groupby(segment_col):
        seg_counts = Counter(item for lst in group["items"] for item in lst)
        share = pd.Series(
            {p: seg_counts.get(p, 0) / len(group) for p in population.index}
        )
        eligible = share >= min_segment_share
        lift = (share / population)[eligible].sort_values(ascending=False).head(n)
        for product, value in lift.items():
            rows.append({
                "segment": seg,
                "product": product,
                "segment_share": round(float(share[product]), 4),
                "population_share": round(float(population[product]), 4),
                "lift": round(float(value), 2),
                "n_baskets": int(seg_counts.get(product, 0)),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Do the two source files actually describe the same behaviour?
# ---------------------------------------------------------------------------
# customer_info gives lifetime spend *per category*; customer_basket gives the
# *items* people put in baskets. The whole promotion story assumes the two
# describe one shopper. This maps products onto the ten spend categories so
# that assumption can be measured instead of assumed.
#
# The mapping is deliberately conservative: only unambiguous products are
# assigned, everything else falls through to 'groceries' (the residual
# category), exactly as the spend schema treats it.
_CATEGORY_PRODUCTS = {
    "electronics": [
        "airpods", "bluetooth headphones", "phone car charger", "phone charger",
        "gadget for tiktok streaming", "iMac", "iPad", "iphone 10", "laptop",
        "ring light", "samsung galaxy 10", "vacuum cleaner",
    ],
    "videogames": [
        "final fantasy XIX", "final fantasy XX", "final fantasy XXII",
        "half-life 2", "half-life: alyx", "megaman zero", "megaman zero 2",
        "megaman zero 3", "megaman zero 4", "metroid fusion", "metroid prime",
        "minecraft", "pokemon scarlet", "pokemon shield", "pokemon sword",
        "pokemon violet", "portal", "portal 2", "ratchet & clank",
        "ratchet & clank 2", "ratchet & clank 3",
    ],
    "hygiene": [
        "body spray", "cologne", "cotton buds", "deodorant", "napkins", "razor",
        "shampoo", "shower gel", "toilet paper", "tooth brush", "toothpaste",
        "water spray",
    ],
    "petfood": ["cat food", "dog food", "pet food"],
    "alcohol_drinks": [
        "beer", "black beer", "champagne", "cider", "dessert wine",
        "french wine", "red wine", "white wine",
    ],
    "nonalcohol_drinks": [
        "antioxydant juice", "black tea", "green tea", "mint green tea",
        "mineral water", "soda", "sparkling water", "tea", "tomato juice",
        "energy drink",
    ],
    "fish": [
        "canned_tuna", "catfish", "fresh tuna", "salmon", "seabass", "shrimp",
        "trout",
    ],
    "meat": [
        "bacon", "burgers", "chicken", "escalope", "ground beef", "ham",
        "hot dogs", "meatballs", "turkey",
    ],
    "vegetables": [
        "asparagus", "avocado", "carrots", "cauliflower", "corn", "eggplant",
        "frozen vegetables", "green beans", "mashed potato", "salad", "shallot",
        "spinach", "tomatoes", "vegetables mix", "yams", "zucchini",
    ],
}
PRODUCT_CATEGORY = {
    product: category
    for category, products in _CATEGORY_PRODUCTS.items()
    for product in products
}
SPEND_CATEGORIES = [c.replace("lifetime_spend_", "") for c in cfg.SPEND_COLUMNS]


def basket_category_shares(baskets: pd.DataFrame) -> pd.DataFrame:
    """Per customer, the share of their basket items falling in each of the
    ten spend categories. Unmapped products count as 'groceries'."""
    per_customer: dict = {}
    for cid, items in zip(baskets["customer_id"].values, baskets["items"].values):
        counter = per_customer.setdefault(cid, Counter())
        for item in items:
            counter[PRODUCT_CATEGORY.get(item, "groceries")] += 1

    out = pd.DataFrame.from_dict(
        {cid: {c: counts.get(c, 0) for c in SPEND_CATEGORIES}
         for cid, counts in per_customer.items()},
        orient="index",
    )
    out = out.div(out.sum(axis=1), axis=0)
    out.index.name = "customer_id"
    return out


def spend_vs_basket_coherence(customers: pd.DataFrame,
                              baskets: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Per category, correlate each customer's lifetime **spend share** against
    the share of their **basket items** in that same category.

    If the two files describe one shopper, these should be clearly positive.
    Rank correlation (Spearman) because both are bounded compositional shares.

    Returns (table, n_customers_with_baskets).
    """
    shares = basket_category_shares(baskets)
    joined = customers.set_index("customer_id").join(
        shares, how="inner", rsuffix="_basket"
    )

    rows = []
    for cat in SPEND_CATEGORIES:
        rho, p = spearmanr(joined[f"share_{cat}"], joined[cat])
        rows.append({
            "category": cat,
            "spearman_rho": round(float(rho), 3),
            "p_value": float(p),
            "mean_spend_share": round(float(joined[f"share_{cat}"].mean()), 4),
            "mean_basket_share": round(float(joined[cat].mean()), 4),
        })
    table = pd.DataFrame(rows).sort_values("spearman_rho").reset_index(drop=True)
    return table, len(joined)


def basket_stats_per_segment(
    baskets_with_segments: pd.DataFrame,
    segment_col: str = "segment",
) -> pd.DataFrame:
    """Number of baskets, average basket size, unique products per segment."""
    rows = []
    for seg, group in baskets_with_segments.groupby(segment_col):
        sizes = group["items"].apply(len)
        unique = len({item for lst in group["items"] for item in lst})
        rows.append(
            {
                "segment": seg,
                "n_baskets": len(group),
                "n_customers": group["customer_id"].nunique(),
                "avg_basket_size": round(sizes.mean(), 2),
                "n_unique_products": unique,
            }
        )
    return pd.DataFrame(rows).sort_values("n_baskets", ascending=False)
