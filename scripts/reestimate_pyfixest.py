"""Independent re-estimation of Tests A and B with pyfixest.

This does not import the project's estimator or analysis code. It rebuilds each sample from
data/processed, fits the models with pyfixest, and compares them with results/*.json.

Needs its own environment: pip install pyfixest pyarrow
Output: results/reestimate_pyfixest.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PROC, RES = ROOT / "data" / "processed", ROOT / "results"
POST, BASE = [2024, 2025, 2026], 2022
QTILES = (0.25, 0.75)


def z(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()


def exposure(t: pd.DataFrame) -> pd.Series:
    return z((z(t["dv_rating_beta"]) + z(t["human_rating_beta"])) / 2)


def gap(x: pd.Series) -> float:
    lo, hi = x.quantile(QTILES[0]), x.quantile(QTILES[1])
    return float(x[x >= hi].mean() - x[x <= lo].mean())


def interact(df: pd.DataFrame, var: str, values: pd.Series, years: list[int]) -> list[str]:
    names = []
    for y in years:
        name = f"{var}_{y}"
        df[name] = values * (df["year"] == y).astype(float)
        names.append(name)
    return names


def contrast(fit, var: str, scale: float, clusters: int) -> dict:
    """Delta = mean(theta for POST) - theta for BASE, with its clustered standard error."""
    names = list(fit.coef().index)
    c = np.zeros(len(names))
    for y in POST:
        c[names.index(f"{var}_{y}")] += 1 / len(POST)
    c[names.index(f"{var}_{BASE}")] -= 1
    b = fit.coef().to_numpy()
    est = float(c @ b) * scale
    se = float(np.sqrt(c @ fit._vcov @ c)) * scale
    q = stats.t.ppf(0.975, clusters - 1)
    return {"est": est, "se": se, "ci95": [est - q * se, est + q * se], "p": float(2 * stats.t.sf(abs(est / se), clusters - 1))}


def pretrend(fit, var: str, years: list[int], clusters: int) -> dict:
    names = list(fit.coef().index)
    ix = [names.index(f"{var}_{y}") for y in years]
    b, v = fit.coef().to_numpy()[ix], fit._vcov[np.ix_(ix, ix)]
    f = float(b @ np.linalg.solve(v, b)) / len(ix)
    return {"F": f, "p": float(stats.f.sf(f, len(ix), clusters - 1))}


def test_a(weighted: bool = False, frame_only: bool = False):
    p = pd.read_parquet(PROC / "ces_panel.parquet")
    p["date"] = pd.to_datetime(p["date"])
    p["year"], p["qtr"] = p["date"].dt.year, p["date"].dt.quarter
    q = p.groupby(["id", "year", "qtr"], as_index=False)["value"].mean()
    q["t"] = q["year"] * 10 + q["qtr"]
    q = q[(q["t"] >= 20151) & (q["t"] <= 20263) & ~((q["t"] >= 20201) & (q["t"] <= 20212))]
    n = q["t"].nunique()
    full = q.groupby("id")["t"].nunique()
    q = q[q["id"].isin(full[full == n].index)].copy()
    t = pd.read_csv(PROC / "ces_industries.csv").set_index("id").loc[sorted(q["id"].unique())]
    x = exposure(t)
    q["y"] = np.log(q["value"])
    years = [y for y in sorted(q["year"].unique()) if y != 2019]
    xs = interact(q, "X", q["id"].map(x), years)
    rs = interact(q, "R", q["id"].map(t["rate_sensitivity"]), years)
    cs = interact(q, "C", q["id"].map(t["cyclicality"]), years)
    if frame_only:
        return q, xs + rs + cs
    kw = {}
    if weighted:
        q["w"] = q["id"].map(t["employed_2015_19"])
        kw["weights"] = "w"
    fit = pf.feols(f"y ~ {' + '.join(xs + rs + cs)} | id + t", q, vcov={"CRV1": "id"}, **kw)
    g = len(t)
    return {"n_industries": g, "n_quarters": n, "quartile_gap_sd": gap(x), "delta": contrast(fit, "X", gap(x), g),
            "pretrend": pretrend(fit, "X", [2015, 2016, 2017, 2018], g)}


def test_a_frame():
    """The Test A regression frame and regressor names, for other checks."""
    return test_a(frame_only=True)


def cells(p: pd.DataFrame, band: str) -> pd.DataFrame:
    d = p[p["age_band"] == band].copy()
    nyears = d["year"].nunique()
    ok = d[d["n_resp_months"] >= 30].groupby(["group", "sector"])["year"].nunique()
    keep = ok[ok == nyears].reset_index()[["group", "sector"]]
    d = d.merge(keep, on=["group", "sector"])
    d["cell"] = d["group"] + "|" + d["sector"]
    w = d[d["year"].between(2015, 2019)].groupby("cell")["employed"].mean()
    d["w"] = d["cell"].map(w)
    d["y"] = np.log(d["employed"])
    return d


def traits() -> pd.DataFrame:
    g = pd.read_csv(PROC / "occ_groups.csv", dtype={"group": str}).set_index("group")
    r = pd.read_csv(PROC / "occ_rate_sensitivity.csv", dtype={"group": str}).set_index("group")
    return g.join(r["cyclicality"])


def test_b(cluster: str = "group") -> dict:
    """cluster: "group" (as reported) or "group+sector" (two-way clustering, a robustness check)."""
    p = pd.read_parquet(PROC / "panel_occ_sector_year.parquet")
    d, t = cells(p, "18-65"), traits()
    x = exposure(t.loc[sorted(d["group"].unique())])
    years = [y for y in sorted(d["year"].unique()) if y != 2019]
    d["sy"] = d["sector"] + "_" + d["year"].astype(str)
    xs = interact(d, "X", d["group"].map(x), years)
    cs = interact(d, "C", d["group"].map(t["cyclicality"]).fillna(0.0), years)
    fit = pf.feols(f"y ~ {' + '.join(xs + cs)} | cell + sy", d, vcov={"CRV1": cluster}, weights="w")
    g = d["group"].nunique() if cluster == "group" else min(d["group"].nunique(), d["sector"].nunique())
    return {"n_cells": d["cell"].nunique(), "n_groups": g, "n_sectors": d["sector"].nunique(), "quartile_gap_sd": gap(x),
            "delta": contrast(fit, "X", gap(x), g), "pretrend": pretrend(fit, "X", list(range(2012, 2019)), g)}


def test_age(cluster: str = "group") -> dict:
    p = pd.read_parquet(PROC / "panel_occ_sector_year.parquet")
    parts = []
    for band, young in [("22-27", 1.0), ("35-65", 0.0)]:
        d = cells(p, band)
        d["young"], d["unit"] = young, d["cell"] + "|" + band
        d["block"] = d["sector"] + "|" + band + "_" + d["year"].astype(str)
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    complete = d.groupby("unit")["year"].nunique()
    d = d[d["unit"].isin(complete[complete == d["year"].nunique()].index)].copy()
    d = d[np.isfinite(d["y"])]
    t = traits()
    x = exposure(t.loc[sorted(d["group"].unique())])
    years = [y for y in sorted(d["year"].unique()) if y != 2019]
    a = interact(d, "X", d["group"].map(x), years)
    b = interact(d, "XY", d["group"].map(x) * d["young"], years)
    fit = pf.feols(f"y ~ {' + '.join(a + b)} | unit + block", d, vcov={"CRV1": cluster}, weights="w")
    g = d["group"].nunique() if cluster == "group" else min(d["group"].nunique(), d["sector"].nunique())
    return {"n_units": d["unit"].nunique(), "n_groups": g, "quartile_gap_sd": gap(x), "delta": contrast(fit, "XY", gap(x), g),
            "pretrend": pretrend(fit, "XY", list(range(2012, 2019)), g)}


def main() -> None:
    a0 = json.loads((RES / "event_study_industry.json").read_text(encoding="utf-8"))
    b0 = json.loads((RES / "event_study_occupation.json").read_text(encoding="utf-8"))
    r0 = json.loads((RES / "review_robustness.json").read_text(encoding="utf-8"))
    out = {"package": f"pyfixest {pf.__version__}", "note": "Written without the project's estimator or analysis code."}
    pairs = [("test_a", test_a(False), a0["primary"]["delta_top_vs_bottom_quartile"], a0["primary"]["pretrend"]),
             ("test_a_weighted", test_a(True), r0["test_a_employment_weighted"], r0["test_a_employment_weighted"]["pretrend"]),
             ("test_b", test_b(), b0["primary"]["delta_top_vs_bottom_quartile"], b0["primary"]["pretrend"]),
             ("young_vs_older", test_age(), b0["age"]["delta_young_minus_old_top_vs_bottom_quartile"], b0["age"]["pretrend"])]
    for name, new, old, pre in pairs:
        out[name] = {**new, "project_delta": {k: old[k] for k in ["est", "se", "ci95", "p"]}, "project_pretrend": {k: pre[k] for k in ["F", "p"]}}
        d = new["delta"]
        print(f"{name}: pyfixest {d['est']*100:+.4f}% se {d['se']*100:.4f} [{d['ci95'][0]*100:+.3f}, {d['ci95'][1]*100:+.3f}] p={d['p']:.5f} F={new['pretrend']['F']:.4f} pre p={new['pretrend']['p']:.5f}")
        print(f"{'':>{len(name)}}  project  {old['est']*100:+.4f}% se {old['se']*100:.4f} [{old['ci95'][0]*100:+.3f}, {old['ci95'][1]*100:+.3f}] p={old['p']:.5f} F={pre['F']:.4f} pre p={pre['p']:.5f}")
        print(f"{'':>{len(name)}}  sample  ", {k: v for k, v in new.items() if k not in ("delta", "pretrend")})
    # Robustness: cluster by occupation group and sector at once; t with (smaller dimension - 1) degrees of freedom.
    out["two_way_clustering"] = {"test_b": test_b("group+sector"), "young_vs_older": test_age("group+sector")}
    for k, v in out["two_way_clustering"].items():
        d = v["delta"]
        print(f"two-way clustered {k}: {d['est']*100:+.2f}% se {d['se']*100:.2f} [{d['ci95'][0]*100:+.2f}, {d['ci95'][1]*100:+.2f}] p={d['p']:.3f}")
    (RES / "reestimate_pyfixest.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
