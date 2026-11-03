"""Test B (docs/PREREGISTRATION.md): inside the same industry, did more AI-exposed occupations
shrink relative to less exposed ones after 2023?

Unit: occupation group x industry sector (CPS, annual, 2011-2026, pandemic months dropped).
Model: cell fixed effects + sector-by-year fixed effects + exposure x year + cyclicality x year.
The sector-by-year effects absorb everything that hit an industry in a year (rates, the
pandemic boom, Section 174, demand), so theta compares occupations only within industries.
Cells need >= 30 employed respondent-months in every year; weighted by 2015-19 employment;
clustered by occupation group (the level exposure varies at).

Headline: Delta = mean(theta 2024-26) - theta 2022, per SD and for most vs least exposed quarter
of occupations. Also: power (ramp injection, first), pre-trend (2012-18), wild bootstrap,
randomization inference, probability below zero, fake 2016 date, dropping computer occupations,
dropping the info/professional sector, removing each cell's own 2011-19 trend, and the age
version (does the gradient hit 22-27 year olds harder than 35-65 year olds in the same cells?).
Writes results/event_study_occupation.json and figures/fig_event_study_occupation.{svg,png}.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import event_study as es  # noqa: E402
from event_study_industry import RAMP_START, REPS, SIZES, SKEPTICAL_PRIOR_SD, r4  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROC, RES, FIG = ROOT / "data" / "processed", ROOT / "results", ROOT / "figures"
MIN_RESP = 30
REF, BASE, POST, PRE = 2019, 2022, [2024, 2025, 2026], [2012, 2013, 2014, 2015, 2016, 2017, 2018]
# Middle of the months each year actually covers, for the injection ramp: 2021 is Jul-Dec,
# 2026 is Jan-Sep. LAST is the end of the data (2026-Q3).
MID = {2021: 2021.75, 2026: 2026.375}
LAST = 2026.75


def load(age_band: str = "18-65") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns log employment (years x cells), each cell's group/sector/weight, and group traits."""
    p = pd.read_parquet(PROC / "panel_occ_sector_year.parquet")
    p = p[p["age_band"] == age_band]
    years = sorted(p["year"].unique())
    ok = p[p["n_resp_months"] >= MIN_RESP].groupby(["group", "sector"])["year"].nunique() == len(years)
    p = p.set_index(["group", "sector"]).loc[ok[ok].index].reset_index()
    p["cell"] = p["group"] + "|" + p["sector"]
    Y = np.log(p.pivot(index="year", columns="cell", values="employed"))
    Y.index = Y.index.astype(float)
    cells = p.drop_duplicates("cell").set_index("cell")[["group", "sector"]]
    cells["weight"] = p[p["year"].between(2015, 2019)].groupby("cell")["employed"].mean()
    traits = pd.read_csv(PROC / "occ_groups.csv", dtype={"group": str}).set_index("group") \
        .join(pd.read_csv(PROC / "occ_rate_sensitivity.csv", dtype={"group": str}).set_index("group")["cyclicality"])
    return Y, cells, traits


def group_exposure(traits: pd.DataFrame, groups) -> pd.Series:
    """Primary exposure across the occupation groups in the sample (1 unit = 1 SD)."""
    t = traits.loc[sorted(set(groups))]
    return es.zscore((es.zscore(t["dv_rating_beta"]) + es.zscore(t["human_rating_beta"])) / 2)


def estimate(Y: pd.DataFrame, cells: pd.DataFrame, xg: pd.Series, traits: pd.DataFrame,
             ref: int = REF, base: int = BASE, post: list[int] = POST, pre: list[int] = PRE) -> tuple[es.Fit, dict]:
    """Fit the Test B model and summarize Delta, its probability below zero, pre-trend and path."""
    c = cells.loc[Y.columns]
    x = c["group"].map(xg)
    cyc = c["group"].map(traits["cyclicality"]).fillna(0.0)    # groups without an estimate: no extra path
    f = es.fit(Y, {"X": x, "C": cyc}, ref, block=c["sector"], cluster=c["group"], weight=c["weight"])
    q = es.quartile_gap(xg)
    con = es.contrast(f, post, base)
    quart = es.test(f, con, q)
    return f, {"n_cells": Y.shape[1], "n_groups": int(c["group"].nunique()), "n_sectors": int(c["sector"].nunique()),
               "quartile_gap_sd": q, "delta_per_sd": es.test(f, con), "delta_top_vs_bottom_quartile": quart,
               "prob_below_zero": es.prob_below_zero(quart["est"], quart["se"]),
               "prob_below_zero_skeptical": es.prob_below_zero(quart["est"], quart["se"], SKEPTICAL_PRIOR_SD),
               "pretrend": es.pretrend(f, pre), "path": es.path(f, ref)}


def randomization_p(Y, cells, xg, traits, est: float, reps: int, rng) -> float:
    """Shuffle exposure across occupation groups; share with |Delta| >= the real |Delta|."""
    hits = 0
    for _ in range(reps):
        perm = pd.Series(rng.permutation(xg.to_numpy()), index=xg.index)
        f, _ = estimate(Y, cells, perm, traits)
        hits += abs(es.contrast(f, POST, BASE) @ f.beta) >= abs(est)
    return hits / reps


def ramp_injection(Y, cells, xg, traits, rng) -> dict:
    """Power first: plant a cut growing linearly from 0 in 2023 to `size` (most vs least exposed
    quarter of groups) at the end of the data, along a shuffled group exposure, and rerun."""
    q = es.quartile_gap(xg)
    mid = np.array([MID.get(int(y), y + 0.5) for y in Y.index])
    ramp = np.clip((mid - RAMP_START) / (LAST - RAMP_START), 0, 1)
    share = float(np.mean([ramp[Y.index == k].mean() for k in POST]))
    perms = [pd.Series(rng.permutation(xg.to_numpy()), index=xg.index) for _ in range(REPS)]
    rows, null = [], None
    for size in SIZES:
        est, rejected = [], []
        for xp in perms:
            planted = np.outer(ramp, cells.loc[Y.columns, "group"].map(xp).to_numpy()) * size / q
            _, r = estimate(Y - planted, cells, xp, traits)
            est.append(r["delta_top_vs_bottom_quartile"]["est"])
            rejected.append(r["delta_top_vs_bottom_quartile"]["p"] < 0.05)
        est = np.array(est)
        null = est if size == 0 else null
        sd0 = float(np.std(null, ddof=1))
        rows.append({"size_at_end": size, "planted_delta": -size * share, "mean_estimate": float(est.mean()),
                     "bias": float(est.mean() + size * share), "null_sd": sd0,
                     "power_empirical_null": float(np.mean(np.abs(est - null.mean()) > 1.96 * sd0)),
                     "power_cluster_test": float(np.mean(rejected))})
        print({k: round(v, 4) for k, v in rows[-1].items()}, flush=True)
    return {"reps": REPS, "planted_share_of_endpoint": share, "rows": rows,
            "mde_80pct_analytic": (1.96 + 0.84) * rows[0]["null_sd"] / share}


def detrend(Y: pd.DataFrame, end: int = 2019) -> pd.DataFrame:
    """Subtract each cell's own straight-line trend, fitted on 2011..end and extended forward."""
    pre = Y[Y.index <= end]
    slope, icept = np.polyfit(pre.index.to_numpy(), pre.to_numpy(), 1)
    return Y - (icept[None, :] + np.outer(Y.index.to_numpy(), slope))


def age_version(traits: pd.DataFrame) -> dict:
    """Young (22-27) and older (35-65) versions of each cell. Effects absorbed: cell x age, and
    sector x age x year. 'XY' = exposure x young: the extra gradient for young workers."""
    Ys, cs = [], []
    for band, young in [("22-27", 1.0), ("35-65", 0.0)]:
        Y, cells, _ = load(band)
        Ys.append(Y.add_suffix(f"|{band}"))
        cs.append(cells.set_axis(cells.index + f"|{band}").assign(young=young, block=(cells["sector"] + f"|{band}").to_numpy()))
    Y, cells = pd.concat(Ys, axis=1).dropna(axis=1), pd.concat(cs)
    cells = cells.loc[Y.columns]
    xg = group_exposure(traits, cells["group"])
    x = cells["group"].map(xg)
    f = es.fit(Y, {"X": x, "XY": x * cells["young"]}, REF, block=cells["block"], cluster=cells["group"],
               weight=cells["weight"])
    q = es.quartile_gap(xg)
    r = es.test(f, es.contrast(f, POST, BASE, "XY"), q)
    return {"n_cells": Y.shape[1], "n_young_cells": int(cells["young"].sum()),
            "delta_young_minus_old_top_vs_bottom_quartile": r,
            "prob_below_zero": es.prob_below_zero(r["est"], r["se"]),
            "pretrend": es.pretrend(f, PRE, "XY"), "path": es.path(f, REF, "XY")}


def plot(main: dict, age: dict, stem: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    blue, muted = "#2a78d6", "#52514e"
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 9))   # stacked, so each panel prints full width
    for ax, r, title in [(axes[0], main, "Within the same industry: more exposed occupations, all ages"),
                         (axes[1], age, "Extra gap for 22–27 year olds vs 35–65 in the same cells")]:
        yrs = np.array(list(r["path"]), float) + 0.5            # plot each year at its midpoint
        th = np.array([v["theta"] for v in r["path"].values()]) * 100
        se = np.array([v["se"] for v in r["path"].values()]) * 100
        gap = np.where(np.diff(yrs) > 1.5)[0] + 1                # break the line across the pandemic gap
        yrs, th, se = (np.insert(a, gap, np.nan) for a in (yrs, th, se))
        ax.fill_between(yrs, th - 1.96 * se, th + 1.96 * se, color=blue, alpha=0.12, lw=0)
        ax.plot(yrs, th, color=blue, lw=2, marker="o", ms=5)
        ax.axhline(0, color=muted, lw=0.8)
        ax.axvspan(2020.0, 2021.5, color="0.95", lw=0)          # pandemic months, dropped
        for x, lab in [(2022.17, "first hike"), (2022.92, "ChatGPT")]:
            ax.axvline(x, color=muted, ls=":", lw=0.8)
        ax.set_title(title, fontsize=13, loc="left")
        ax.grid(axis="y", color="0.9", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=11)
        ax.set_ylabel("Employment gap per 1 SD more exposure\n(%, relative to 2019; 95% CI shaded)", fontsize=11)
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=150)


def main() -> None:
    rng = np.random.default_rng(2026)
    Y, cells, traits = load()
    xg = group_exposure(traits, cells["group"])
    out = {"spec": "docs/PREREGISTRATION.md, Test B", "min_respondent_months": MIN_RESP}

    print("ramp injection test (power first)...", flush=True)
    out["ramp_injection"] = ramp_injection(Y, cells, xg, traits, rng)

    f, out["primary"] = estimate(Y, cells, xg, traits)
    con = es.contrast(f, POST, BASE)
    out["primary"]["wild_bootstrap_p"] = es.wild_bootstrap_p(f, con, 999, rng)
    out["primary"]["randomization_p"] = randomization_p(Y, cells, xg, traits, float(con @ f.beta), 1000, rng)

    # Pre-set checks. Fake date: pretend the shock came in 2016, compared with 2015, 2011-19 data.
    early = Y[Y.index <= 2019]
    out["fake_2016"] = estimate(early, cells, xg, traits, post=[2016, 2017, 2018], base=2015, pre=[2011, 2012, 2013])[1]
    keep = cells.index[cells["group"] != "15-12"].intersection(Y.columns)
    out["drop_computer_occupations"] = estimate(Y[keep], cells, group_exposure(traits, cells.loc[keep, "group"]), traits)[1]
    keep = cells.index[cells["sector"] != "info_professional"].intersection(Y.columns)
    out["drop_info_professional"] = estimate(Y[keep], cells, group_exposure(traits, cells.loc[keep, "group"]), traits)[1]
    out["own_trend_removed"] = estimate(detrend(Y), cells, xg, traits)[1]
    out["age"] = age_version(traits)

    out = r4(out)
    (RES / "event_study_occupation.json").write_text(json.dumps(out, indent=2))
    plot(out["primary"], out["age"], FIG / "fig_event_study_occupation")
    for k in ["primary", "fake_2016", "drop_computer_occupations", "drop_info_professional", "own_trend_removed"]:
        r = out[k]
        print(f"{k}: {r['n_cells']} cells, Delta (quartile) {r['delta_top_vs_bottom_quartile']}, "
              f"P(<0) {r['prob_below_zero']}, pre-trend p {r['pretrend']['p']}")
    a = out["age"]
    print(f"age: {a['n_cells']} cells, Delta young-old {a['delta_young_minus_old_top_vs_bottom_quartile']}, "
          f"P(<0) {a['prob_below_zero']}, pre-trend p {a['pretrend']['p']}")
    print("wrote results/event_study_occupation.json")


if __name__ == "__main__":
    main()
