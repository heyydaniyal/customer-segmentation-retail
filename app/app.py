"""
Streamlit demo for the retail customer segmentation project.

Reads only precomputed aggregates from app/data/ (built by prep_app_data.py)
plus the persisted models in outputs/models/. No customer-level data is
loaded by the app.

Run locally from the repo root:  streamlit run app/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg          # noqa: E402
from src import clustering as clu      # noqa: E402
from src import preprocessing as pp    # noqa: E402
from app import batch                   # noqa: E402

APP_DATA = ROOT / "app" / "data"

st.set_page_config(page_title="Retail Customer Segmentation",
                   layout="centered")

# ---------------------------------------------------------------------------
# Visual identity: one muted palette, one color per segment, used everywhere.
# ---------------------------------------------------------------------------
SEGMENT_COLORS = {
    "Engaged Generalist":         "#2f6f64",  # deep teal  (the core base)
    "Routine Generalist":         "#7fb0a5",  # soft teal
    "Promo Hunter":               "#c98a3b",  # warm ochre (stands apart)
    "Routine Young Parents":      "#5a8fb0",  # muted blue
    "Routine Tech":               "#4b6b8a",  # slate blue
    "Routine Family Provisioner": "#a88b6a",  # taupe
    "Engaged Family Provisioner": "#8a6d8f",  # dusty mauve
}
INK = "#1f2933"
MUTED = "#6b7780"
GRID = "#e6ebe9"


def seg_color(name: str) -> str:
    return SEGMENT_COLORS.get(name, "#9aa5a1")


CUSTOM_CSS = """
<style>
/* tighten the top padding Streamlit leaves */
.block-container { padding-top: 2.2rem; max-width: 820px; }

/* header block */
.app-header { margin-bottom: 0.4rem; }
.app-header h1 {
    font-size: 1.9rem; font-weight: 700; letter-spacing: -0.02em;
    color: #1f2933; margin: 0 0 0.15rem 0;
}
.app-header .subtitle { color: #6b7780; font-size: 1.02rem; margin: 0; }
.stat-strip {
    display: flex; gap: 1.6rem; margin: 0.9rem 0 0.4rem 0;
    padding-bottom: 0.9rem; border-bottom: 1px solid #e6ebe9;
}
.stat-strip .stat { display: flex; flex-direction: column; }
.stat-strip .stat .num {
    font-size: 1.25rem; font-weight: 700; color: #2f6f64; line-height: 1;
}
.stat-strip .stat .lbl {
    font-size: 0.74rem; color: #6b7780; text-transform: uppercase;
    letter-spacing: 0.04em; margin-top: 0.2rem;
}

/* metric cards */
div[data-testid="stMetric"] {
    background: #f5f7f6; border: 1px solid #e6ebe9;
    border-radius: 10px; padding: 0.7rem 0.9rem;
}
div[data-testid="stMetricLabel"] p { color: #6b7780; font-size: 0.78rem; }

/* segment badge */
.seg-badge {
    display: inline-block; padding: 0.32rem 0.8rem; border-radius: 999px;
    color: #fff; font-weight: 600; font-size: 0.98rem; margin: 0.2rem 0 0.4rem 0;
}

/* tabs: a little more air, accent underline */
button[data-baseweb="tab"] { font-size: 0.98rem; }

/* dataframe: lighter */
[data-testid="stDataFrame"] { border-radius: 8px; }
</style>
"""

# ---------------------------------------------------------------------------
# Segment notes.
# ---------------------------------------------------------------------------
SEGMENT_NOTES = {
    "Engaged Generalist": (
        "The biggest segment and the biggest spenders. Very broad baskets "
        "and a long history with the retailer - not the record on either "
        "(the small Engaged Family Provisioner group edges them out), but "
        "at 8,300 customers this is the core of the business. What ties "
        "their baskets together is breakfast staples."
    ),
    "Routine Generalist": (
        "Similar grocery-heavy mix to the engaged group but about two-fifths "
        "of the basket breadth. They shop as often as the engaged group, so "
        "the opportunity is more categories per basket, not more visits."
    ),
    "Promo Hunter": (
        "71% of their purchases are on promotion. No other segment is even "
        "above 32%. You're not going to change that behaviour; the "
        "promotion designs just channel it somewhere useful."
    ),
    "Routine Young Parents": (
        "The centroid points to vegetables and hygiene, but open the actual "
        "baskets and it's pet food, baby food, napkins, cooking oil. The name "
        "follows the baskets: kids at home are about average for the base. "
        "Highest purchase frequency of any segment."
    ),
    "Routine Tech": (
        "40% of spend goes to electronics. Fewest stores visited, smallest "
        "households. It produces the strongest association rule in the data "
        "(bluetooth headphones to airpods, lift 4.0) - though by basket lift "
        "its real signature is personal care, five times the population rate."
    ),
    "Routine Family Provisioner": (
        "Spending tilts to meat, fish and pet food; promo use is low at "
        "18%. Their actual baskets over-index on tech "
        "accessories instead - the two source files disagree here, and the "
        "repo documents why rather than papering over it."
    ),
    "Engaged Family Provisioner": (
        "Small group but punches above its weight. Most distinct products "
        "of any segment, longest tenure, largest households."
    ),
}
METRIC_LABELS = {
    "total_spend": ("Lifetime spend", "{:,.0f}"),
    "promo_pct": ("Bought on promotion", "{:.0%}"),
    "lifetime_total_distinct_products": ("Distinct products", "{:,.0f}"),
    "distinct_stores_visited": ("Stores visited", "{:.1f}"),
    "tenure_years": ("Tenure (years)", "{:.1f}"),
}


@st.cache_data
def load_app_data():
    profiles = pd.read_csv(APP_DATA / "segment_profiles.csv", index_col=0)
    top_products = pd.read_csv(APP_DATA / "top_products.csv")
    rules = pd.read_csv(APP_DATA / "segment_rules_top.csv")
    designs = pd.read_csv(APP_DATA / "promotion_designs.csv")
    defaults = json.loads((APP_DATA / "form_defaults.json").read_text())
    presets = json.loads((APP_DATA / "presets.json").read_text())
    return profiles, top_products, rules, designs, defaults, presets


@st.cache_resource
def load_models():
    return clu.load_models()


profiles, top_products, rules, designs, defaults, presets = load_app_data()
segments_by_size = (profiles.drop(index="POPULATION")
                    .sort_values("n_customers", ascending=False))
population = profiles.loc["POPULATION"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
HEATMAP_COLS = {
    "total_spend": "lifetime spend",
    "promo_pct": "promo share",
    "lifetime_total_distinct_products": "distinct products",
    "distinct_stores_visited": "stores visited",
    "tenure_years": "tenure",
    "share_groceries": "groceries %",
    "share_electronics": "electronics %",
    "share_vegetables": "vegetables %",
    "share_hygiene": "hygiene %",
    "share_meat": "meat %",
}


@st.cache_data
def segment_heatmap():
    data = segments_by_size[list(HEATMAP_COLS)].rename(columns=HEATMAP_COLS)
    z = (data - data.mean()) / data.std()
    fig, ax = plt.subplots(figsize=(8, 3.2))
    im = ax.imshow(z.values, cmap="BuGn", aspect="auto",
                   vmin=-2.2, vmax=2.2)
    ax.set_xticks(range(len(z.columns)))
    ax.set_xticklabels(z.columns, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(z.index)))
    ax.set_yticklabels(z.index, fontsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="z-score")
    fig.tight_layout()
    return fig


def nearest_segments(feats: dict) -> list[tuple[str, float]]:
    """Rank all (value, mix) centroid combinations by joint distance in the
    two standardised spaces and return the distinct final segments in order.
    K-Means gives hard assignments, not probabilities; this reports honest
    centroid distances instead."""
    models = load_models()
    row = pd.DataFrame([feats])
    dv = models["kmeans_v"].transform(
        models["scaler_v"].transform(row[cfg.VALUE_FEATURES].values))[0]
    dm = models["kmeans_p"].transform(
        models["scaler_p"].transform(row[cfg.PRODUCT_MIX_FEATURES].values))[0]

    folded = models.get("folded", {})
    combos = []
    for vi, v_dist in enumerate(dv):
        for mi, m_dist in enumerate(dm):
            v_name = models["value_map"][vi]
            m_name = models["mix_map"][mi]
            seg = clu.FINAL_SEGMENT_LOOKUP[(v_name, m_name)]
            seg = folded.get(seg, seg)
            combos.append((seg, float(v_dist + m_dist)))

    combos.sort(key=lambda t: t[1])
    seen, ranked = set(), []
    for seg, dist in combos:
        if seg not in seen:
            seen.add(seg)
            ranked.append((seg, dist))
    return ranked


def apply_preset(name: str) -> None:
    p = presets[name]
    for col, val in p["spends"].items():
        st.session_state[f"sp_{col}"] = float(val)
    st.session_state["in_products"] = int(p["distinct_products"])
    st.session_state["in_stores"] = int(p["distinct_stores"])
    st.session_state["in_tenure"] = float(p["tenure_years"])
    st.session_state["in_promo"] = float(p["promo_pct"])


def _style_axes(ax):
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.set_axisbelow(True)


def segment_size_chart(sizes: pd.Series):
    fig, ax = plt.subplots(figsize=(8, 3.0))
    colors = [seg_color(s) for s in sizes.index]
    ax.barh(range(len(sizes)), sizes.values, color=colors, height=0.68)
    ax.set_yticks(range(len(sizes)))
    ax.set_yticklabels(sizes.index)
    ax.invert_yaxis()
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    for i, v in enumerate(sizes.values):
        ax.text(v + max(sizes) * 0.01, i, f"{v:,}", va="center",
                fontsize=8.5, color=MUTED)
    ax.set_xlim(0, max(sizes) * 1.12)
    _style_axes(ax)
    ax.tick_params(axis="x", labelbottom=False)
    fig.tight_layout()
    return fig


def shares_chart(shares: pd.Series, color: str):
    fig, ax = plt.subplots(figsize=(8, 3.3))
    ax.barh(range(len(shares)), shares.values, color=color, height=0.7)
    ax.set_yticks(range(len(shares)))
    ax.set_yticklabels(shares.index)
    ax.invert_yaxis()
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    _style_axes(ax)
    fig.tight_layout()
    return fig


def dist_chart(dist: pd.Series):
    fig, ax = plt.subplots(figsize=(8, 2.8))
    colors = [seg_color(s) for s in dist.index]
    ax.barh(range(len(dist)), dist.values, color=colors, height=0.68)
    ax.set_yticks(range(len(dist)))
    ax.set_yticklabels(dist.index)
    ax.invert_yaxis()
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    _style_axes(ax)
    fig.tight_layout()
    return fig


def badge(seg: str):
    st.markdown(
        f'<span class="seg-badge" style="background:{seg_color(seg)}">'
        f'{seg}</span>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_segments():
    st.write(
        "33,038 customers of a Lisbon retailer, segmented by two independent "
        "K-Means clusterings (engagement and product mix) combined through "
        "an explicit business-rule layer. Details and code in the "
        "[repository](https://github.com/heyydaniyal/customer-segmentation-retail)." 
    )

    sizes = segments_by_size["n_customers"].astype(int)
    st.markdown("**Segment sizes**")
    st.pyplot(segment_size_chart(sizes))

    st.markdown("**All segments at a glance** — each metric standardised "
                "across segments (dark = above the segment average, "
                "light = below)")
    st.pyplot(segment_heatmap())

    st.divider()
    seg = st.selectbox("Explore a segment", sizes.index.tolist(),
                       key="seg_explore")
    row = profiles.loc[seg]

    badge(seg)
    st.write(SEGMENT_NOTES[seg])
    st.caption(f"{int(row['n_customers']):,} customers "
               f"({row['n_customers'] / population['n_customers']:.0%} of the base)")

    cols = st.columns(len(METRIC_LABELS))
    for col, (key, (label, fmt)) in zip(cols, METRIC_LABELS.items()):
        delta = (row[key] - population[key]) / population[key]
        col.metric(label, fmt.format(row[key]), f"{delta:+.0%} vs avg",
                   delta_color="off")

    st.markdown("**Where the money goes** — share of lifetime spend by category")
    shares = (row[cfg.PRODUCT_MIX_FEATURES]
              .rename(lambda c: c.replace("share_", "").replace("_", " "))
              .sort_values(ascending=False))
    st.pyplot(shares_chart(shares, seg_color(seg)))

    st.markdown("**Most common basket items**")
    tp = top_products[top_products["segment"] == seg]
    st.dataframe(
        tp[["product", "baskets", "pct_of_baskets"]]
        .rename(columns={"pct_of_baskets": "% of baskets"})
        .set_index("product"),
        width='stretch', height=280,
    )


def init_form_state() -> None:
    """Seed the classify form once with population medians; presets then
    overwrite these keys. Widgets read state via key only, which avoids
    Streamlit's default-value-vs-session-state warning."""
    if "in_promo" in st.session_state:
        return
    for col, val in defaults["spend_medians"].items():
        st.session_state[f"sp_{col}"] = float(val)
    st.session_state["in_products"] = int(defaults["distinct_products"])
    st.session_state["in_stores"] = int(defaults["distinct_stores"])
    st.session_state["in_tenure"] = float(defaults["tenure_years"])
    st.session_state["in_promo"] = float(defaults["promo_pct"])


def form_to_features(spends, n_products, n_stores, tenure, promo) -> dict:
    """Run the form inputs through the same frozen preprocessing pipeline
    that batch scoring uses (imputation medians and 99th-percentile spend
    caps included), so the form and an uploaded file always agree on a
    customer's segment. Fields the form does not ask for are left blank and
    filled by the frozen medians; none of them feed the two perspectives."""
    raw = {
        "customer_id": 0, "customer_name": "Form Customer",
        "customer_gender": "female", "customer_birthdate": None,
        "kids_home": 0.0, "teens_home": 0.0, "number_complaints": 0.0,
        "typical_hour": np.nan, "loyalty_card_number": np.nan,
        "latitude": np.nan, "longitude": np.nan,
        "distinct_stores_visited": float(n_stores),
        "lifetime_total_distinct_products": float(n_products),
        "percentage_of_products_bought_promotion": float(promo),
        "year_first_transaction": float(cfg.MAX_VALID_FIRST_TRANSACTION_YEAR - tenure),
        **{k: float(v) for k, v in spends.items()},
    }
    row = pp.build_features(pd.DataFrame([raw]),
                            load_models()["preproc_params"]).iloc[0]
    return row[cfg.VALUE_FEATURES + cfg.PRODUCT_MIX_FEATURES].to_dict()


def page_classify():
    init_form_state()
    st.write(
        "Enter a customer's history and the persisted models classify them "
        "into a segment. This ranks every centroid combination by distance, "
        "which picks the same winner as the pipeline's `predict_segment` but "
        "also shows you the runner-up. Start from a preset or from the "
        "population medians."
    )

    preset_cols = st.columns(len(presets))
    for col, name in zip(preset_cols, presets):
        col.button(name, on_click=apply_preset, args=(name,),
                   width="stretch")

    with st.expander("Spend by category (lifetime)", expanded=True):
        spend_cols = st.columns(2)
        spends = {}
        for i, col in enumerate(cfg.SPEND_COLUMNS):
            label = col.replace("lifetime_spend_", "").replace("_", " ")
            spends[col] = spend_cols[i % 2].number_input(
                label, min_value=0.0, step=50.0, key=f"sp_{col}",
            )

    c1, c2 = st.columns(2)
    n_products = c1.number_input("Distinct products ever bought", 1, 1000,
                                 key="in_products")
    n_stores = c2.number_input("Distinct stores visited", 1, 20,
                               key="in_stores")
    tenure = c1.number_input("Years as a customer", 0.0, 40.0, step=0.5,
                             key="in_tenure")
    promo = c2.slider("Share of purchases on promotion", 0.0, 1.0,
                      key="in_promo")

    if st.button("Classify", type="primary"):
        total = sum(spends.values())
        if total <= 0:
            st.error("Total spend must be positive.")
            return
        feats = form_to_features(spends, n_products, n_stores, tenure, promo)

        ranked = nearest_segments(feats)
        seg = ranked[0][0]

        badge(seg)
        st.write(SEGMENT_NOTES[seg])
        if len(ranked) > 1:
            st.caption(
                f"Next-closest centroid combination: {ranked[1][0]}. "
                "K-Means assigns hard labels, so distances are reported "
                "instead of probabilities."
            )

        row = profiles.loc[seg]
        st.caption("How this customer compares with the segment's average:")
        comp = pd.DataFrame({
            "this customer": [total, promo, n_products, n_stores, tenure],
            "segment average": [row["total_spend"], row["promo_pct"],
                                row["lifetime_total_distinct_products"],
                                row["distinct_stores_visited"],
                                row["tenure_years"]],
        }, index=["lifetime spend", "promo share", "distinct products",
                  "stores visited", "tenure (years)"])
        st.dataframe(comp.round(2), width="stretch")

    st.caption(
        "Models trained on this retailer's (synthetic) data; the "
        "classification is illustrative, not a general-purpose scorer."
    )


def page_promotions():
    st.write(
        "Association rules mined per segment (Apriori; every rule must "
        "appear in at least 30 baskets of its segment). Each promotion "
        "design cites the rule that justifies it."
    )

    seg = st.selectbox("Segment", segments_by_size.index.tolist(),
                       key="seg_promo")

    design = designs[designs["segment"] == seg]
    if not design.empty:
        d = design.iloc[0]
        st.subheader("Proposed campaign")
        st.write(d["campaign_idea"])
        st.markdown(f"**Why this works:** {d['logic']}")
        st.caption(f"Evidence: {d['evidence']}")

    st.subheader("Top association rules")
    seg_rules = rules[rules["segment"] == seg]
    if seg_rules.empty:
        st.write("Basket evidence too thin for reliable rules in this "
                 "segment; its promotion is profile-driven (see above).")
    else:
        st.dataframe(
            seg_rules.drop(columns="segment")
            .rename(columns={"antecedents": "if the basket has",
                             "consequents": "it also has",
                             "n_baskets": "baskets"})
            .set_index("if the basket has"),
            width='stretch', height=350,
        )


def page_batch():
    st.write(
        "Upload a customer CSV in the same schema as `customer_info.csv` "
        "and every row is classified through the full pipeline "
        "(preprocessing, both clusterings, segment assignment). "
        "Download the sample file to see the expected format."
    )
    sample_bytes = (APP_DATA / "sample_batch.csv").read_bytes()
    n_sample = sample_bytes.decode().count("\n") - 1
    st.download_button(
        f"Download sample file ({n_sample} customers)",
        sample_bytes, file_name="sample_batch.csv", mime="text/csv",
    )

    upload = st.file_uploader("Customer CSV", type=["csv"])
    if upload is None:
        return

    with st.spinner("Scoring..."):
        result, error = batch.score_batch(upload.getvalue(),
                                          models=load_models())
    if error:
        st.error(error)
        return

    st.write(f"Scored {len(result):,} customers.")
    dist = result["segment"].value_counts()
    st.pyplot(dist_chart(dist))
    st.dataframe(result.head(20).set_index("customer_id"),
                 width="stretch", height=280)
    st.download_button(
        "Download results",
        result.to_csv(index=False).encode(),
        file_name="customer_segments_scored.csv", mime="text/csv",
        type="primary",
    )
    st.caption(
        "Imputation medians and outlier caps are frozen from the training "
        "population, not re-derived from your file, so a customer scores the "
        "same whether you upload them alone or inside a full export."
    )


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown(
    """
    <div class="app-header">
      <h1>Retail Customer Segmentation</h1>
      <p class="subtitle">Two-perspective clustering with a business-rule
      overlay, and per-segment promotion designs.</p>
    </div>
    <div class="stat-strip">
      <div class="stat"><span class="num">33,038</span>
        <span class="lbl">customers</span></div>
      <div class="stat"><span class="num">7</span>
        <span class="lbl">segments</span></div>
      <div class="stat"><span class="num">100k</span>
        <span class="lbl">basket transactions</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Segments", "Classify a customer", "Promotions", "Batch scoring"])
with tab1:
    page_segments()
with tab2:
    page_classify()
with tab3:
    page_promotions()
with tab4:
    page_batch()
