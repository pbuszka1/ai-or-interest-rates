"""Pipeline checks for docs/VERIFICATION_AUDIT.md: crosswalk coverage, partial-year divisors, the triple-difference
panel, log conversion, quartile gaps, and an export of the SDiD matrices for a reference-package comparison.

Output: results/verification_checks.json and data/processed/sdid_reference_input_*.csv
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_exposure_crosswalk as bx  # noqa: E402
import event_study as es  # noqa: E402
import event_study_occupation as eo  # noqa: E402
import sdid_industry as si  # noqa: E402
import young_worker_robustness as yw  # noqa: E402
from synth_industry import PANDEMIC, load  # noqa: E402

PROC, RES = ROOT / "data" / "processed", ROOT / "results"


def unmatched() -> dict:
    """Which Census occupation codes have no exposure score, by share of 2020-25 employment."""
    cs = bx.census_scores(bx.soc_scores(), bx.census_to_soc())
    m = bx.census_to_occ2010().merge(cs, on="census2018", how="left")
    tot = m["weight"].sum()
    miss = m[m["dv_rating_beta"].isna()].groupby("census2018")["weight"].sum().sort_values(ascending=False)
    titles = bx.census_to_soc().drop_duplicates("census2018").set_index("census2018")["title"]
    return {"share_unmatched": float(miss.sum() / tot),
            "top10": [{"census2018": int(k) if str(k).isdigit() else k, "title": str(titles.get(k, ""))[:60], "share": float(v / tot)} for k, v in miss.head(10).items()]}


def soc_collapse() -> dict:
    """How O*NET 8-digit scores become 6-digit SOC scores."""
    src = (ROOT / "scripts" / "build_exposure_crosswalk.py").read_text(encoding="utf-8")
    body = src[src.index("def soc_scores"):src.index("def ", src.index("def soc_scores") + 10)]
    return {"code": body.strip()[:900], "uses_drop_duplicates": "drop_duplicates" in body}


def divisors() -> dict:
    p = pd.read_parquet(PROC / "panel_occ_sector_year.parquet")
    months = p.drop_duplicates("year").set_index("year")["months"].astype(int).to_dict()
    cps = ds.dataset(ROOT / "data" / "interim" / "cps", format="parquet", partitioning="hive")
    actual = {}
    for y in [2019, 2021, 2025, 2026]:
        t = cps.to_table(columns=["MONTH"], filter=ds.field("YEAR") == y).to_pandas()
        actual[y] = sorted(t["MONTH"].unique().tolist())
    a = p[p["age_band"] == "18-65"]
    keep = a[a["n_resp_months"] >= eo.MIN_RESP].groupby(["group", "sector"])["year"].nunique()
    fails = a[a["n_resp_months"] < eo.MIN_RESP].groupby("year").size().to_dict()
    return {"divisor_by_year": {int(k): v for k, v in months.items()}, "months_present": {k: v for k, v in actual.items()},
            "cells_failing_30_by_year": {int(k): int(v) for k, v in fails.items()},
            "cells_kept": int((keep == a["year"].nunique()).sum())}


def triple_difference_panel() -> dict:
    raw = pd.read_parquet(PROC / "panel_occ_sector_year.parquet")
    rs = pd.read_csv(PROC / "occ_rate_sensitivity.csv", dtype={"group": str}).set_index("group")
    traits = pd.read_csv(PROC / "occ_groups.csv", dtype={"group": str}).set_index("group").join(rs[["cyclicality"]])
    ys, cs = [], []
    for band, young in [("22-27", 1.0), ("35-65", 0.0)]:
        y, c = yw.select(raw, band, "every_year_30")
        ys.append(y.add_suffix(f"|{band}"))
        cs.append(c.set_axis(c.index + f"|{band}").assign(young=young, block=(c["sector"] + f"|{band}").to_numpy(), cell=c.index))
    Y, C = pd.concat(ys, axis=1).dropna(axis=1), pd.concat(cs)
    C = C.loc[Y.columns]
    young_cells = set(C.loc[C.young == 1, "cell"])
    older_cells = set(C.loc[C.young == 0, "cell"])
    xg = eo.group_exposure(traits, C["group"])
    q = es.quartile_gap(xg)
    x = C["group"].map(xg)
    f = es.fit(Y, {"X": x, "XY": x * C["young"]}, eo.REF, block=C["block"], cluster=C["group"], weight=C["weight"])
    as_written = es.test(f, es.contrast(f, eo.POST, eo.BASE, "XY"), q)
    # Cell-by-year effects: keep cells with both ages, then difference young minus old within each cell and year.
    both = sorted(young_cells & older_cells)
    yo = pd.DataFrame({c: Y[f"{c}|22-27"] - Y[f"{c}|35-65"] for c in both})
    cc = C[C.young == 1].set_index("cell").loc[both]
    xb = cc["group"].map(xg)
    out = {}
    for label, w in [("weights_young_cell", cc["weight"]), ("weights_equal", pd.Series(1.0, index=both))]:
        fb = es.fit(yo, {"XY": xb}, eo.REF, block=cc["sector"], cluster=cc["group"], weight=w)
        out[label] = es.test(fb, es.contrast(fb, eo.POST, eo.BASE, "XY"), q)
    w = C["weight"]
    return {"young_cells": len(young_cells), "older_cells": len(older_cells), "young_cells_with_older_cell": len(both),
            "weights": "each cell-by-age series is weighted by its own 2015-19 employment in that age band",
            "weight_share_young": float(w[C.young == 1].sum() / w.sum()),
            "as_written": as_written, "cell_by_year_effects_difference_model": out}


def log_conversion() -> dict:
    hits = {}
    for f in sorted((ROOT / "scripts").glob("*.py")):
        t = f.read_text(encoding="utf-8")
        if re.search(r"np\.exp\(|math\.exp\(|expm1", t):
            hits[f.name] = [ln.strip()[:100] for ln in t.splitlines() if re.search(r"np\.exp\(|math\.exp\(|expm1", ln)]
    return {"files_using_exp": hits}


def quartile_gaps() -> dict:
    a = json.loads((RES / "event_study_industry.json").read_text(encoding="utf-8"))
    b = json.loads((RES / "event_study_occupation.json").read_text(encoding="utf-8"))
    y = json.loads((RES / "young_worker_robustness.json").read_text(encoding="utf-8"))
    src = (ROOT / "scripts" / "event_study.py").read_text(encoding="utf-8")
    return {"test_a": a["primary"]["quartile_gap_sd"], "test_b": b["primary"]["quartile_gap_sd"], "triple_difference": y["young_vs_older"]["quartile_gap_sd"],
            "definition": re.search(r"def quartile_gap.*?return[^\n]*", src, re.S).group(0), "standard_normal_value": 2 * 1.2711}


def export_sdid() -> dict:
    emp, t = load()
    g = si.groups(t, emp)
    out = {}
    for name, pre, post, drop in [("2022_high_pre_ends_2019", (2003.0, 2019.75), (2022.0, 2026.5), (2020.0, 2021.75)),
                                  ("2022_high", (2003.0, 2021.75), (2022.0, 2026.5), PANDEMIC)]:
        w, T_pre = si.window(emp, pre, post, drop)
        ids = list(g["high"]) + list(g["controls"])
        long = w[ids].stack().rename("outcome").reset_index()
        long.columns = ["time", "unit", "outcome"]
        long["treatment"] = (long["unit"].isin(g["high"]) & (long["time"] >= post[0])).astype(int)
        long.to_csv(PROC / f"sdid_reference_input_{name}.csv", index=False)
        out[name] = {"T_pre": int(T_pre), "periods": int(w.shape[0]), "treated": len(g["high"]), "controls": len(g["controls"])}
    return out


def main() -> None:
    out = {"unmatched_occupations": unmatched(), "soc_collapse": soc_collapse(), "partial_years": divisors(),
           "triple_difference": triple_difference_panel(), "log_conversion": log_conversion(), "quartile_gaps": quartile_gaps(),
           "sdid_export": export_sdid()}
    (RES / "verification_checks.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(json.dumps(out, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
