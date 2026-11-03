"""Exploratory, not pre-registered: how much does the young-worker estimate depend on specification?

Reruns the CPS triple difference from Test B under alternative choices, and adds Test B checks.
The January-to-September versions rebuild the panel from the CPS microdata, so they need
data/interim/cps (not redistributed). The others need only data/processed.

Output: results/young_worker_robustness.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import event_study as es  # noqa: E402
import event_study_occupation as eo  # noqa: E402
from build_occ_sector_panel import AGE_BANDS, COLUMNS, sector  # noqa: E402
from build_panel import EMPLOYED, occ2010_to_group  # noqa: E402

PROC, CPS = ROOT / "data" / "processed", ROOT / "data" / "interim" / "cps"
POST, BASE, REF, PRE = eo.POST, eo.BASE, eo.REF, eo.PRE
PER_MONTH = eo.MIN_RESP / 12


def select(p: pd.DataFrame, band: str, rule: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """rule: every_year_30 (as pre-registered), every_year_per_month, or through_2019."""
    p = p[p["age_band"] == band].copy()
    years = sorted(p["year"].unique())
    if rule == "every_year_30":
        p["ok"], use = p["n_resp_months"] >= eo.MIN_RESP, years
    elif rule == "every_year_per_month":
        p["ok"], use = p["n_resp_months"] / p["months"] >= PER_MONTH, years
    else:
        p["ok"], use = p["n_resp_months"] >= eo.MIN_RESP, [y for y in years if y <= 2019]
    ok = p[p["ok"] & p["year"].isin(use)].groupby(["group", "sector"])["year"].nunique() == len(use)
    q = p.set_index(["group", "sector"]).loc[ok[ok].index].reset_index()
    q = q[q["employed"] > 0]
    full = q.groupby(["group", "sector"])["year"].nunique() == len(years)  # the log needs a value every year
    q = q.set_index(["group", "sector"]).loc[full[full].index].reset_index()
    q["cell"] = q["group"] + "|" + q["sector"]
    y = np.log(q.pivot(index="year", columns="cell", values="employed"))
    y.index = y.index.astype(float)
    c = q.drop_duplicates("cell").set_index("cell")[["group", "sector"]]
    c["weight"] = q[q["year"].between(2015, 2019)].groupby("cell")["employed"].mean()
    return y, c


def age_fit(p, traits, rule="every_year_30", controls=False, detrended=False):
    ys, cs = [], []
    for band, young in [("22-27", 1.0), ("35-65", 0.0)]:
        y, c = select(p, band, rule)
        ys.append(y.add_suffix(f"|{band}"))
        cs.append(c.set_axis(c.index + f"|{band}").assign(young=young, block=(c["sector"] + f"|{band}").to_numpy()))
    y, c = pd.concat(ys, axis=1).dropna(axis=1), pd.concat(cs)
    c = c.loc[y.columns]
    y = eo.detrend(y) if detrended else y
    xg = eo.group_exposure(traits, c["group"])
    x = c["group"].map(xg)
    t = {"X": x, "XY": x * c["young"]}
    if controls:
        cyc = c["group"].map(traits["cyclicality"]).fillna(0.0)
        rate = c["group"].map(traits["rate_sensitivity_raw"]).fillna(0.0)
        t.update({"C": cyc, "CY": cyc * c["young"], "R": rate, "RY": rate * c["young"]})
    f = es.fit(y, t, REF, block=c["block"], cluster=c["group"], weight=c["weight"])
    return f, es.quartile_gap(xg), y.shape[1], int((c["young"] == 1).sum())


def b_fit(p, traits, rule="every_year_30", raw_rate=False):
    y, c = select(p, "18-65", rule)
    xg = eo.group_exposure(traits, c["group"])
    t = {"X": c["group"].map(xg), "C": c["group"].map(traits["cyclicality"]).fillna(0.0)}
    if raw_rate:
        t["R"] = c["group"].map(traits["rate_sensitivity_raw"]).fillna(0.0)
    f = es.fit(y, t, REF, block=c["sector"], cluster=c["group"], weight=c["weight"])
    return f, es.quartile_gap(xg), y.shape[1]


def from_ref(f, var, q):
    c = np.zeros(len(f.names))
    for k in POST:
        c[f.names.index(f"{var}:{k}")] = 1 / len(POST)
    return es.test(f, c, q)


def slope(f, var):
    d = np.array([k - REF for k in PRE], float)
    c = np.zeros(len(f.names))
    for k, dk in zip(PRE, d):
        c[f.names.index(f"{var}:{k}")] = dk / np.sum(d ** 2)
    return es.test(f, c)


def january_to_september() -> pd.DataFrame:
    groups = occ2010_to_group()
    data = ds.dataset(CPS, format="parquet", partitioning="hive")
    parts = []
    for year in range(2011, 2027):
        df = data.to_table(columns=COLUMNS, filter=ds.field("YEAR") == year).to_pandas()
        df = df[~((df["YEAR"] == 2020) | ((df["YEAR"] == 2021) & (df["MONTH"] <= 6)))]
        df = df[df["MONTH"] <= 9]
        if df.empty:
            continue
        months = df["MONTH"].nunique()
        df = df[df["EMPSTAT"].isin(EMPLOYED)].merge(groups, on="OCC2010", how="inner")
        df["sector"] = sector(df["IND1990"])
        df = df.dropna(subset=["sector"])
        for band, (lo, hi) in AGE_BANDS.items():
            d = df[df["AGE"].between(lo, hi)]
            g = d.groupby(["group", "sector"]).agg(employed=("WTFINL", "sum"), n_resp_months=("WTFINL", "size"))
            g["employed"] /= months
            parts.append(g.reset_index().assign(year=year, age_band=band, months=months))
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    raw = pd.read_parquet(PROC / "panel_occ_sector_year.parquet")
    rs = pd.read_csv(PROC / "occ_rate_sensitivity.csv", dtype={"group": str}).set_index("group")
    traits = pd.read_csv(PROC / "occ_groups.csv", dtype={"group": str}).set_index("group") \
        .join(rs[["cyclicality", "rate_sensitivity", "rate_sensitivity_raw"]])
    out = {"note": "Exploratory, not pre-registered, not corrected for multiple testing.",
           "occupational_rate_sensitivity_sd_after_shrinkage": float(rs["rate_sensitivity"].std()),
           "occupational_rate_sensitivity_sd_raw": float(rs["rate_sensitivity_raw"].std())}

    f, q, n, ny = age_fit(raw, traits)
    young = {"quartile_gap_sd": q, "reported": {"cells": n, "young_cells": ny, **es.test(f, es.contrast(f, POST, BASE, "XY"), q)},
             "excluding_2026": es.test(f, es.contrast(f, [2024, 2025], BASE, "XY"), q),
             "relative_to_2019": from_ref(f, "XY", q),
             "pretrend_omnibus": es.pretrend(f, PRE, "XY"),
             "pretrend_linear_slope_per_year_per_sd": slope(f, "XY"),
             "theta_2021_2022_equal_2019": es.pretrend(f, [2021, 2022], "XY")}
    for name, kw in [("every_year_per_month", {"rule": "every_year_per_month"}), ("filter_through_2019", {"rule": "through_2019"}),
                     ("age_interacted_cyclicality_and_rates", {"controls": True}), ("cell_trends_removed", {"detrended": True})]:
        f, q, n, ny = age_fit(raw, traits, **kw)
        young[name] = {"cells": n, "young_cells": ny, **es.test(f, es.contrast(f, POST, BASE, "XY"), q)}
    # Same cells only: young minus older within each occupation-sector cell and year, which absorbs cell-by-year
    # effects. Eq. 5 instead compares 167 young cells with 484 older cells, most of them different.
    yb, cb = select(raw, "22-27", "every_year_30")
    ob, _ = select(raw, "35-65", "every_year_30")
    both = [c_ for c_ in yb.columns if c_ in ob.columns]
    diff = yb[both] - ob[both]
    cc = cb.loc[both]
    xg = eo.group_exposure(traits, cc["group"])
    q = es.quartile_gap(xg)
    for name, w in [("same_cells_cell_by_year", cc["weight"]), ("same_cells_cell_by_year_equal_weights", pd.Series(1.0, index=both))]:
        f = es.fit(diff, {"XY": cc["group"].map(xg)}, REF, block=cc["sector"], cluster=cc["group"], weight=w)
        young[name] = {"cells": len(both), **es.test(f, es.contrast(f, POST, BASE, "XY"), q), "pretrend": es.pretrend(f, PRE, "XY")}
    out["young_vs_older"] = young

    f, q, n = b_fit(raw, traits)
    out["test_b"] = {"quartile_gap_sd": q, "reported": es.test(f, es.contrast(f, POST, BASE), q), "relative_to_2019": from_ref(f, "X", q),
                     "real_se_mde_on_estimand": 2.8 * es.test(f, es.contrast(f, POST, BASE), q)["se"],
                     "pretrend_linear_slope_per_year_per_sd": slope(f, "X"),
                     "theta_2021_2022_equal_2019": es.pretrend(f, [2021, 2022])}
    f, q, n = b_fit(raw, traits, raw_rate=True)
    out["test_b"]["with_raw_occupational_rate_sensitivity"] = es.test(f, es.contrast(f, POST, BASE), q)

    if CPS.exists():
        js = january_to_september()
        f, q, n, ny = age_fit(js, traits, rule="every_year_per_month")
        out["young_vs_older"]["january_to_september"] = {"cells": n, "young_cells": ny, **es.test(f, es.contrast(f, POST, BASE, "XY"), q)}
        f, q, n = b_fit(js, traits, rule="every_year_per_month")
        out["test_b"]["january_to_september"] = es.test(f, es.contrast(f, POST, BASE), q)

    path = ROOT / "results" / "young_worker_robustness.json"
    path.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    for k, v in young.items():
        if isinstance(v, dict) and "est" in v:
            print(f"{k}: {v['est']*100:+.1f} [{v['ci95'][0]*100:+.1f}, {v['ci95'][1]*100:+.1f}] p={v['p']:.3f}")


if __name__ == "__main__":
    main()
