"""
Precompute everything the Streamlit app needs, so the app itself never
touches customer-level data.

Also re-fits and re-saves the two KMeans models (plus the frozen
preprocessing parameters) with the *installed* scikit-learn version. Joblib files pickled under an older sklearn can fail
to load on Streamlit Cloud; re-fitting is safe here because the clustering
is seed-stable (pairwise ARI > 0.99 across seeds, see notebook 03) and we
assert below that the refit reproduces the shipped segment assignment
exactly before overwriting anything.

Run from the repo root:  python app/prep_app_data.py
"""

from __future__ import annotations

import ast
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg                      # noqa: E402
from src import clustering as clu                  # noqa: E402
from src import preprocessing as pp                # noqa: E402
from app import batch                              # noqa: E402

APP_DATA = ROOT / "app" / "data"
APP_DATA.mkdir(parents=True, exist_ok=True)

PROFILE_COLS = [
    "total_spend", "promo_pct", "lifetime_total_distinct_products",
    "distinct_stores_visited", "tenure_years", "age",
    "kids_home", "teens_home",
] + cfg.PRODUCT_MIX_FEATURES


def refit_and_save_models(df: pd.DataFrame, params: dict) -> pd.Series:
    """Re-fit both perspectives, verify against the shipped segments,
    then persist with the current sklearn."""
    scaler_v, Xv = clu.scale_perspective(df, cfg.VALUE_FEATURES)
    km_v, labels_v = clu.fit_kmeans(Xv, n_clusters=3)
    scaler_p, Xp = clu.scale_perspective(df, cfg.PRODUCT_MIX_FEATURES)
    km_p, labels_p = clu.fit_kmeans(Xp, n_clusters=4)

    work = df.copy()
    work["value_cluster"] = labels_v
    work["mix_cluster"] = labels_p
    segments, naming = clu.assign_final_segments(work)

    shipped = pd.read_csv(cfg.OUTPUT_DIR / "customer_segments.csv")
    shipped = shipped.set_index("customer_id")["segment"]
    refit = pd.Series(segments.values, index=df["customer_id"].values)
    match = (refit.reindex(shipped.index) == shipped).mean()
    print(f"Refit vs shipped segment agreement: {match:.4%}")
    if match != 1.0:
        raise RuntimeError(
            "Refit does not reproduce the shipped segmentation - investigate "
            "before overwriting the models."
        )

    clu.save_models(scaler_v, km_v, scaler_p, km_p,
                    naming["value_map"], naming["mix_map"],
                    naming["folded"], preproc_params=params)
    print("Models re-saved with the installed scikit-learn version.")
    work["segment"] = segments
    return work


def build_profiles(work: pd.DataFrame) -> None:
    prof = work.groupby("segment")[PROFILE_COLS].mean()
    prof["n_customers"] = work["segment"].value_counts()
    prof.loc["POPULATION"] = work[PROFILE_COLS].mean().tolist() + [len(work)]
    prof.round(4).to_csv(APP_DATA / "segment_profiles.csv")
    print(f"segment_profiles.csv: {prof.shape}")


def build_defaults(work: pd.DataFrame) -> None:
    """Population medians used to prefill the classify form."""
    defaults = {
        "spend_medians": {
            col: round(float(work[col].median()), 2)
            for col in cfg.SPEND_COLUMNS
        },
        "distinct_products": int(work["lifetime_total_distinct_products"].median()),
        "distinct_stores": int(work["distinct_stores_visited"].median()),
        "tenure_years": round(float(work["tenure_years"].median()), 1),
        "promo_pct": round(float(work["promo_pct"].median()), 3),
    }
    (APP_DATA / "form_defaults.json").write_text(json.dumps(defaults, indent=2))
    print("form_defaults.json written")


PRESET_PERSONAS = {
    # button label -> segment whose medians define the persona
    "Deal chaser": "Promo Hunter",
    "Electronics buyer": "Routine Tech",
    "Young family": "Routine Young Parents",
    "Loyal big-basket shopper": "Engaged Generalist",
}


def build_presets(work: pd.DataFrame) -> None:
    """One-click personas for the classify form, built from real segment
    medians. Each preset is pushed through predict_segment before being
    written, so a preset can never demo a wrong answer."""
    import numpy as np

    models = clu.load_models()
    presets = {}
    for label, seg in PRESET_PERSONAS.items():
        grp = work[work["segment"] == seg]
        preset = {
            "spends": {col: round(float(grp[col].median()), 2)
                       for col in cfg.SPEND_COLUMNS},
            "distinct_products": int(grp["lifetime_total_distinct_products"].median()),
            "distinct_stores": int(grp["distinct_stores_visited"].median()),
            "tenure_years": round(float(grp["tenure_years"].median()), 1),
            "promo_pct": round(float(grp["promo_pct"].median()), 3),
        }

        total = sum(preset["spends"].values())
        feats = {
            "total_spend_log": np.log1p(total),
            "lifetime_total_distinct_products": preset["distinct_products"],
            "distinct_stores_visited": preset["distinct_stores"],
            "tenure_years": preset["tenure_years"],
            "promo_pct": preset["promo_pct"],
        }
        for col, val in preset["spends"].items():
            feats[f"share_{col.replace('lifetime_spend_', '')}"] = val / total
        got = clu.predict_segment(pd.DataFrame([feats]), models=models)
        got_seg = got["segment"].iloc[0]
        if got_seg != seg:
            raise RuntimeError(
                f"Preset '{label}' classifies as {got_seg}, expected {seg} - "
                "adjust the persona before shipping."
            )
        print(f"  preset '{label}' -> {got_seg} (verified)")
        presets[label] = preset

    (APP_DATA / "presets.json").write_text(json.dumps(presets, indent=2))
    print("presets.json written")


def build_top_products(work: pd.DataFrame) -> None:
    baskets = pd.read_csv(cfg.DATA_DIR / "customer_basket.csv")
    seg_map = work.set_index("customer_id")["segment"]
    baskets["segment"] = baskets["customer_id"].map(seg_map)

    rows = []
    for seg, grp in baskets.groupby("segment"):
        counter: Counter = Counter()
        for goods in grp["list_of_goods"]:
            counter.update(ast.literal_eval(goods))
        n_baskets = len(grp)
        for product, count in counter.most_common(12):
            rows.append({
                "segment": seg, "product": product, "baskets": count,
                "pct_of_baskets": round(100 * count / n_baskets, 1),
            })
    pd.DataFrame(rows).to_csv(APP_DATA / "top_products.csv", index=False)
    print("top_products.csv written")


def build_sample_batch(work: pd.DataFrame, n: int = 15) -> None:
    """
    Regenerate the downloadable template for the batch-scoring tab.

    It has to be in the *raw* customer_info schema, because that is what users
    upload — including customer_name (the degree prefix is parsed out of it)
    and customer_birthdate (age is derived from it). So the columns stay, but
    the values are replaced with obvious placeholders: no file shipped in this
    repo should carry rows that read like real people, and a template least of
    all. The degree prefixes are preserved so the file still exercises that
    branch of the pipeline.

    Generated here rather than hand-maintained so it can never drift from the
    schema batch.REQUIRED_COLUMNS enforces.
    """
    raw = pd.read_csv(cfg.RAW_CUSTOMER_INFO)

    # One customer from each segment where possible, so the template
    # demonstrates a spread of results rather than 15 near-identical rows.
    seg_of = work.set_index("customer_id")["segment"]
    picks: list[int] = []
    for _, group in raw.assign(segment=raw["customer_id"].map(seg_of)).groupby("segment"):
        picks.extend(group["customer_id"].head(2).tolist())
    picks = picks[:n]
    sample = raw[raw["customer_id"].isin(picks)].copy().reset_index(drop=True)

    degrees = sample["customer_name"].str.extract(r"^(Bsc|Msc|Phd)\.", expand=False)
    sample["customer_name"] = [
        f"{d}. Sample Customer {i:02d}" if isinstance(d, str) else f"Sample Customer {i:02d}"
        for i, d in enumerate(degrees, start=1)
    ]

    # Keep exactly the columns the scorer requires. Extra columns are
    # tolerated by score_batch, but a template should show the minimum, and
    # lat/lon are location data with no role in the pipeline.
    sample = sample[batch.REQUIRED_COLUMNS]

    out_path = APP_DATA / "sample_batch.csv"
    sample.to_csv(out_path, index=False)

    scored, error = batch.score_batch(out_path.read_bytes())
    if error:
        raise RuntimeError(f"the generated sample file does not score: {error}")
    print(f"sample_batch.csv: {len(sample)} rows, "
          f"{scored['segment'].nunique()} distinct segments, scores cleanly")


def slim_rules() -> None:
    rules = pd.read_csv(cfg.OUTPUT_DIR / "segment_rules.csv")
    keep = ["segment", "antecedents", "consequents",
            "support", "confidence", "lift", "n_baskets"]
    slim = (rules[keep]
            .sort_values(["segment", "lift"], ascending=[True, False])
            .groupby("segment").head(25))
    slim.round(3).to_csv(APP_DATA / "segment_rules_top.csv", index=False)
    print(f"segment_rules_top.csv: {len(slim)} rules")


def main() -> None:
    df = pd.read_parquet(cfg.OUTPUT_DIR / "customer_info_processed.parquet")
    # The preprocessing params are a training-time artefact; refit them from
    # the raw file so they are persisted alongside the re-saved models.
    params = pp.fit_preprocessing_params(pp.load_customer_info())
    work = refit_and_save_models(df, params)
    build_profiles(work)
    build_defaults(work)
    build_presets(work)
    build_top_products(work)
    build_sample_batch(work)
    slim_rules()
    # promotion designs are already aggregate-level; copy as-is
    designs = pd.read_csv(cfg.OUTPUT_DIR / "promotion_designs.csv")
    designs.to_csv(APP_DATA / "promotion_designs.csv", index=False)
    print("promotion_designs.csv copied")
    print("\nApp data ready in app/data/")


if __name__ == "__main__":
    main()
