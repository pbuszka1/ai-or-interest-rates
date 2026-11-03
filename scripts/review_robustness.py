"""Exploratory, not pre-registered: checks prompted by an outside referee review.

Test A under employment weights and other occupation mixes, Tests B and the young-worker version with an education control,
permutation inference within education groups, what Test B's interval implies for a gradual ramp, and how
the GPT-4 and human ratings correlate. Needs data/interim/cps (not redistributed).

Output: results/review_robustness.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_industry_panel as bip  # noqa: E402
import event_study as es  # noqa: E402
import event_study_industry as ei  # noqa: E402
import event_study_occupation as eo  # noqa: E402
import young_worker_robustness as yw  # noqa: E402
from build_panel import EMPLOYED, occ2010_to_group  # noqa: E402

PROC, CPS = ROOT / "data" / "processed", ROOT / "data" / "interim" / "cps"
POST, BASE, REF, PRE = eo.POST, eo.BASE, eo.REF, eo.PRE
REPS = 1000
COLLEGE_MIN, EDUC_MISSING = 111, 999  # IPUMS EDUC: 111 is a bachelor's degree


def college_share() -> pd.Series:
    """Share of each occupation group's 2015-19 employment holding a bachelor's degree or more."""
    data = ds.dataset(CPS, format="parquet", partitioning="hive")
    d = data.to_table(columns=["EDUC", "EMPSTAT", "OCC2010", "WTFINL", "AGE"],
                      filter=(ds.field("YEAR") >= 2015) & (ds.field("YEAR") <= 2019)).to_pandas()
    d = d[d["EMPSTAT"].isin(EMPLOYED) & d["AGE"].between(18, 65)].merge(occ2010_to_group(), on="OCC2010")
    d["college"] = d["EDUC"].between(COLLEGE_MIN, EDUC_MISSING - 1)
    return d.groupby("group").apply(lambda v: np.average(v["college"], weights=v["WTFINL"]), include_groups=False)


def test_a(mix: tuple[int, int], emp, base) -> dict:
    e = bip.industry_exposure(bip.leaves(), *mix).set_index("id")
    t = base.copy()
    ids = t.index.intersection(e.index)
    cols = ["dv_rating_beta", "human_rating_beta"]
    corr = float(np.corrcoef(t.loc[ids, cols[0]], e.loc[ids, cols[0]])[0, 1])
    t.loc[ids, cols] = e.loc[ids, cols]
    _, r, _, _ = ei.run(emp, t, ei.PRIMARY)
    return {"correlation_with_primary_score": corr, "estimate": r["delta_top_vs_bottom_quartile"], "pretrend": r["pretrend"]}


def main() -> None:
    out = {"note": "Exploratory, not pre-registered, not corrected for multiple testing."}

    x = pd.read_csv(PROC / "exposure_occ2010.csv")
    groups = pd.read_csv(PROC / "occ_groups.csv", dtype={"group": str}).set_index("group")
    cov = np.cov(x["dv_rating_beta"], x["human_rating_beta"], aweights=x["employment_weight"].fillna(0))
    out["rating_correlation"] = {"occupations": float(x["dv_rating_beta"].corr(x["human_rating_beta"])),
                                 "occupations_employment_weighted": float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])),
                                 "occupation_groups": float(groups["dv_rating_beta"].corr(groups["human_rating_beta"]))}

    emp, base = ei.load()
    out["test_a_other_mixes"] = {"2020_25": test_a((2020, 2025), emp, base), "2020_22": test_a((2020, 2022), emp, base)}

    rs = pd.read_csv(PROC / "occ_rate_sensitivity.csv", dtype={"group": str}).set_index("group")
    traits = groups.join(rs[["cyclicality", "rate_sensitivity_raw"]])
    share = college_share().reindex(traits.index)
    traits["college"] = es.zscore(share)
    raw = pd.read_parquet(PROC / "panel_occ_sector_year.parquet")

    y, c = yw.select(raw, "18-65", "every_year_30")
    xg = eo.group_exposure(traits, c["group"])
    q = es.quartile_gap(xg)
    t0 = {"X": c["group"].map(xg), "C": c["group"].map(traits["cyclicality"]).fillna(0.0)}
    college = c["group"].map(traits["college"]).fillna(0.0)
    rate = c["group"].map(traits["rate_sensitivity_raw"]).fillna(0.0)

    def fit_b(t):
        return es.fit(y, t, REF, block=c["sector"], cluster=c["group"], weight=c["weight"])

    f0 = fit_b(t0)
    f1, f2 = fit_b(t0 | {"E": college}), fit_b(t0 | {"E": college, "R": rate})
    reported = es.test(f0, es.contrast(f0, POST, BASE), q)
    out["test_b"] = {
        "exposure_college_correlation": float(xg.corr(traits["college"].reindex(xg.index))),
        "reported": reported,
        "with_college_by_year": {**es.test(f1, es.contrast(f1, POST, BASE), q), "pretrend": es.pretrend(f1, PRE)},
        "with_college_and_raw_rate": {**es.test(f2, es.contrast(f2, POST, BASE), q), "pretrend": es.pretrend(f2, PRE)}}

    share_of_endpoint = json.loads((ROOT / "results" / "event_study_occupation.json").read_text(encoding="utf-8"))["ramp_injection"]["planted_share_of_endpoint"]
    out["test_b"]["interval_as_linear_ramp_endpoint"] = {"planted_share_of_endpoint": share_of_endpoint,
                                                         "ci95": [v / share_of_endpoint for v in reported["ci95"]]}

    rng = np.random.default_rng(20261102)
    est0 = float(es.contrast(f0, POST, BASE) @ f0.beta)
    strata = pd.qcut(traits["college"].reindex(xg.index), 4, labels=False)
    perms = {}
    for label, s in [("unrestricted", pd.Series(0, index=xg.index)), ("within_college_quartile", strata)]:
        hits = 0
        for _ in range(REPS):
            perm = xg.copy()
            for k in s.unique():
                ix = s.index[s == k]
                perm.loc[ix] = rng.permutation(xg.loc[ix].to_numpy())
            fp = fit_b({"X": c["group"].map(perm), "C": t0["C"]})
            hits += abs(es.contrast(fp, POST, BASE) @ fp.beta) >= abs(est0)
        perms[label] = hits / REPS
    out["test_b"]["permutation_p"] = perms

    ys, cs = [], []
    for band, young in [("22-27", 1.0), ("35-65", 0.0)]:
        yy, cc = yw.select(raw, band, "every_year_30")
        ys.append(yy.add_suffix(f"|{band}"))
        cs.append(cc.set_axis(cc.index + f"|{band}").assign(young=young, block=(cc["sector"] + f"|{band}").to_numpy()))
    ya, ca = pd.concat(ys, axis=1).dropna(axis=1), pd.concat(cs)
    ca = ca.loc[ya.columns]
    xa = eo.group_exposure(traits, ca["group"])
    qa = es.quartile_gap(xa)
    xx = ca["group"].map(xa)
    ea = ca["group"].map(traits["college"]).fillna(0.0)
    fy = es.fit(ya, {"X": xx, "XY": xx * ca["young"], "E": ea, "EY": ea * ca["young"]}, REF, block=ca["block"], cluster=ca["group"], weight=ca["weight"])
    out["young_vs_older_with_college_by_age_by_year"] = {**es.test(fy, es.contrast(fy, POST, BASE, "XY"), qa), "pretrend": es.pretrend(fy, PRE, "XY")}

    # Pre-registered robustness check for Test A: employment weights.
    Y = ei.panel(emp, ei.PRIMARY)
    ids = list(Y.columns)
    w = base.loc[ids, "employed_2015_19"]
    xa2 = ei.exposure(base, ids)
    spec = ei.PRIMARY
    q2 = es.quartile_gap(xa2)
    ctl = ei.controls(base, ids)

    def fit_a(keep, extra=None):
        return es.fit(Y[keep], {"X": xa2.loc[keep]} | {k: v.loc[keep] for k, v in ctl.items()} | (extra or {}), spec.ref_year, weight=w.loc[keep])

    fa = fit_a(ids)
    ca2 = es.contrast(fa, spec.post_years, spec.base_year)
    rng2 = np.random.default_rng(2026)
    wild = es.wild_bootstrap_p(fa, ca2, 999, rng2)
    arr, hits = xa2.to_numpy(), 0
    for _ in range(REPS):
        fp = es.fit(Y, {"X": pd.Series(rng2.permutation(arr), index=ids)} | ctl, spec.ref_year, weight=w)
        hits += abs(es.contrast(fp, spec.post_years, spec.base_year) @ fp.beta) >= abs(float(ca2 @ fa.beta))
    fb = fit_a(ids, {"B": ei.boom(emp, ids)})
    loo = []
    for i in ids:
        keep = [k for k in ids if k != i]
        fk = fit_a(keep)
        loo.append(es.test(fk, es.contrast(fk, spec.post_years, spec.base_year), q2)["est"])

    def other(sp):
        yy2 = ei.panel(emp, sp)
        keep = [c for c in yy2.columns if pd.notna(base.loc[c, "employed_2015_19"])]
        xx2 = ei.exposure(base, keep)
        fo = es.fit(yy2[keep], {"X": xx2} | ei.controls(base, keep), sp.ref_year, weight=base.loc[keep, "employed_2015_19"])
        return {**es.test(fo, es.contrast(fo, sp.post_years, sp.base_year), es.quartile_gap(xx2)), "pretrend": es.pretrend(fo, sp.pre_years), "n_industries": len(keep)}

    out["test_a_employment_weighted"] = {
        **es.test(fa, ca2, q2), "pretrend": es.pretrend(fa, spec.pre_years), "wild_bootstrap_p": wild, "randomization_p": hits / REPS,
        "effective_industries": float(w.sum() ** 2 / (w ** 2).sum()), "leave_one_out_range": [min(loo), max(loo)],
        "boom_controlled": {**es.test(fb, es.contrast(fb, spec.post_years, spec.base_year), q2), "pretrend": es.pretrend(fb, spec.pre_years)},
        "dotcom": other(ei.DOTCOM), "fake_2017": other(ei.FAKE)}

    # Plan differences: cyclicality in the age version, and an occupation-level rather than cell-level trend.
    cyc_a = ca["group"].map(traits["cyclicality"]).fillna(0.0)
    kw = {"block": ca["block"], "cluster": ca["group"], "weight": ca["weight"]}
    fc1 = es.fit(ya, {"X": xx, "XY": xx * ca["young"], "C": cyc_a}, REF, **kw)
    out["young_vs_older_with_common_cyclicality"] = es.test(fc1, es.contrast(fc1, POST, BASE, "XY"), qa)
    out["young_vs_older_cell_counts"] = {"young": int((ca["young"] == 1).sum()), "older": int((ca["young"] == 0).sum())}
    slope, _ = np.polyfit(y[y.index <= 2019].index.to_numpy(), y[y.index <= 2019].to_numpy(), 1)
    slopes = pd.Series(slope, index=y.columns)
    gs = slopes.groupby(c.loc[y.columns, "group"]).transform(lambda v: np.average(v, weights=c.loc[v.index, "weight"]))
    yd = y - np.outer(y.index.to_numpy() - 2019, gs.to_numpy())
    ft = es.fit(yd, t0, REF, block=c["sector"], cluster=c["group"], weight=c["weight"])
    out["test_b"]["occupation_level_trend_removed"] = es.test(ft, es.contrast(ft, POST, BASE), q)

    path = ROOT / "results" / "review_robustness.json"
    path.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(json.dumps(out, indent=1, default=float)[:2500])


if __name__ == "__main__":
    main()
