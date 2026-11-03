"""Test A (docs/PREREGISTRATION.md): does more AI exposure predict a bigger employment fall
across all CES industries after 2023?

Data: quarterly log CES payroll employment, 2015-Q1 to 2026-Q3, pandemic quarters (2020-Q1 to
2021-Q2) dropped, industries observed in every quarter. 2019 is the reference year.
Exposure X: mean of the z-scored GPT-4 and human beta ratings, re-standardized (1 unit = 1 SD).
Controls: rate sensitivity and cyclicality, each with its own year-by-year path.

Headline: Delta = mean(theta 2024-26) - theta 2022, i.e. how much the exposure gradient moved
from "rate hikes, no ChatGPT" to "AI adoption". Reported per SD and for the most vs least
exposed quarter of industries, with clustered, wild-bootstrap and randomization p-values and
the probability the effect is below zero.

Checks: power (ramp injection, run first), pre-trend, boom-size control, dot-com placebo,
fake 2017 date, exposure quintiles, and (exploratory) the extrapolated pre-trend.
Writes results/event_study_industry.json and figures/fig_event_study_industry.{svg,png}.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import event_study as es  # noqa: E402
from synth_industry import PANDEMIC, load  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES, FIG = ROOT / "results", ROOT / "figures"
LAST = 2026.5                                 # last quarter in the data, 2026-Q3
SIZES, REPS, RAMP_START = [0.0, 0.025, 0.05, 0.10, 0.15], 200, 2023.0
SKEPTICAL_PRIOR_SD = 0.025                    # prior for P(effect < 0); see the paper


@dataclass
class Spec:
    window: tuple[float, float]
    ref_year: int
    base_year: int          # the "rates, no ChatGPT" year subtracted in Delta
    post_years: list[int]   # the "adoption" years averaged in Delta
    pre_years: list[int]    # tested jointly for a pre-trend


PRIMARY = Spec((2015.0, LAST), 2019, 2022, [2024, 2025, 2026], [2015, 2016, 2017, 2018])
DOTCOM = Spec((1996.0, 2003.75), 1998, 2001, [2002, 2003], [1996, 1997])
FAKE = Spec((2010.0, 2019.75), 2015, 2016, [2017, 2018], [2010, 2011, 2012, 2013])


def panel(emp: pd.DataFrame, spec: Spec) -> pd.DataFrame:
    """Quarters x industries inside the window, pandemic dropped, complete industries only."""
    q = emp[(emp.index >= spec.window[0]) & (emp.index <= spec.window[1])]
    q = q[~((q.index >= PANDEMIC[0]) & (q.index <= PANDEMIC[1]))]
    return q.loc[:, q.notna().all()]


def exposure(t: pd.DataFrame, ids) -> pd.Series:
    return es.zscore((es.zscore(t.loc[ids, "dv_rating_beta"]) + es.zscore(t.loc[ids, "human_rating_beta"])) / 2)


def controls(t: pd.DataFrame, ids) -> dict[str, pd.Series]:
    return {"R": t.loc[ids, "rate_sensitivity"], "C": t.loc[ids, "cyclicality"]}


def boom(emp: pd.DataFrame, ids) -> pd.Series:
    """How far 2022-Q2 log employment sat above each industry's own 2015-19 straight-line trend."""
    pre = emp.loc[(emp.index >= 2015.0) & (emp.index <= 2019.75), ids]
    slope, icept = np.polyfit(pre.index.to_numpy(), pre.to_numpy(), 1)
    return pd.Series(emp.loc[2022.25, ids].to_numpy() - (icept + slope * 2022.25), index=ids)


def summarize(f: es.Fit, spec: Spec, x: pd.Series, n: tuple[int, int]) -> dict:
    """Delta per SD and for top vs bottom quartile, pre-trend test, theta path."""
    c, q = es.contrast(f, spec.post_years, spec.base_year), es.quartile_gap(x)
    quart = es.test(f, c, q)
    p = es.path(f, spec.ref_year)
    return {"n_quarters": n[0], "n_industries": n[1], "quartile_gap_sd": q,
            "delta_per_sd": es.test(f, c), "delta_top_vs_bottom_quartile": quart,
            "prob_below_zero": es.prob_below_zero(quart["est"], quart["se"]),
            "prob_below_zero_skeptical": es.prob_below_zero(quart["est"], quart["se"], SKEPTICAL_PRIOR_SD),
            "theta_base_year": p[spec.base_year]["theta"],
            "theta_post_mean": float(np.mean([p[k]["theta"] for k in spec.post_years])),
            "pretrend": es.pretrend(f, spec.pre_years), "path": p}


def run(emp, t, spec: Spec, extra: dict | None = None) -> tuple[es.Fit, dict, pd.DataFrame, pd.Series]:
    Y = panel(emp, spec)
    ids = list(Y.columns)
    x = exposure(t, ids)
    f = es.fit(Y, {"X": x} | controls(t, ids) | (extra or {}), spec.ref_year)
    return f, summarize(f, spec, x, Y.shape), Y, x


def randomization_p(Y: pd.DataFrame, t: pd.DataFrame, est: float, reps: int, rng) -> float:
    """Shuffle exposure across industries; share of shuffles with |Delta| >= the real |Delta|."""
    ids = list(Y.columns)
    x = exposure(t, ids).to_numpy()
    hits = 0
    for _ in range(reps):
        f = es.fit(Y, {"X": pd.Series(rng.permutation(x), index=ids)} | controls(t, ids), PRIMARY.ref_year)
        hits += abs(es.contrast(f, PRIMARY.post_years, PRIMARY.base_year) @ f.beta) >= abs(est)
    return hits / reps


def ramp_injection(emp, t, rng) -> dict:
    """Power first. Plant a cut that grows in a straight line from 0 in 2023-Q1 to `size` (most vs
    least exposed quarter) in 2026-Q3, along a SHUFFLED exposure score (so it is unrelated to the
    real one), rerun the primary model, and record how often it's detected."""
    Y = panel(emp, PRIMARY)
    ids = list(Y.columns)
    x = exposure(t, ids).to_numpy()
    q = es.quartile_gap(pd.Series(x))
    tt = Y.index.to_numpy()
    ramp = np.clip((tt - RAMP_START) / (LAST - RAMP_START), 0, 1)
    year = np.floor(tt).astype(int)
    share = float(np.mean([ramp[year == k].mean() for k in PRIMARY.post_years]))  # planted Delta / size
    perms = [rng.permutation(x) for _ in range(REPS)]
    rows, null = [], None
    for size in SIZES:
        est, rejected = [], []
        for xp in perms:
            f = es.fit(Y - np.outer(ramp, xp) * size / q, {"X": pd.Series(xp, index=ids)} | controls(t, ids),
                       PRIMARY.ref_year)
            r = es.test(f, es.contrast(f, PRIMARY.post_years, PRIMARY.base_year), q)
            est.append(r["est"])
            rejected.append(r["p"] < 0.05)
        est = np.array(est)
        null = est if size == 0 else null
        sd0 = float(np.std(null, ddof=1))
        rows.append({"size_at_2026Q3": size, "planted_delta": -size * share, "mean_estimate": float(est.mean()),
                     "bias": float(est.mean() + size * share), "null_sd": sd0,
                     "power_empirical_null": float(np.mean(np.abs(est - null.mean()) > 1.96 * sd0)),
                     "power_cluster_test": float(np.mean(rejected))})
        print({k: round(v, 4) for k, v in rows[-1].items()}, flush=True)
    powered = [r["size_at_2026Q3"] for r in rows
               if r["size_at_2026Q3"] > 0 and min(r["power_empirical_null"], r["power_cluster_test"]) >= 0.8]
    return {"reps": REPS, "ramp": f"linear from {RAMP_START} to {LAST}", "planted_share_of_endpoint": share,
            "rows": rows, "mde_80pct_analytic": (1.96 + 0.84) * rows[0]["null_sd"] / share,
            "smallest_tested_size_with_80pct_power": min(powered) if powered else None}


def plot(main: dict, boomed: dict, dot: dict, stem: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    blue, orange, muted = "#2a78d6", "#eb6834", "#52514e"
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 9), sharey=True)   # stacked, so each panel prints full width
    panels = [(axes[0], [(main, blue, "Primary (rate + cyclicality controls)"), (boomed, orange, "Plus boom-size control")],
               "2015–2026: does more AI exposure mean a bigger fall?"),
              (axes[1], [(dot, blue, "Same model, dot-com era")], "1996–2003 placebo (no generative AI)")]
    for ax, series, title in panels:
        for j, (r, col, lab) in enumerate(series):
            yrs = np.array(list(r["path"])) + 0.5 + 0.08 * j
            th = np.array([v["theta"] for v in r["path"].values()]) * 100
            se = np.array([v["se"] for v in r["path"].values()]) * 100
            gap = np.where(np.diff(yrs) > 1.5)[0] + 1          # break the line across the dropped pandemic year
            yrs, th, se = (np.insert(a, gap, np.nan) for a in (yrs, th, se))
            if j == 0:
                ax.fill_between(yrs, th - 1.96 * se, th + 1.96 * se, color=col, alpha=0.12, lw=0)
            ax.plot(yrs, th, color=col, lw=2, marker="o", ms=5, label=lab)
        ax.axhline(0, color=muted, lw=0.8)
        ax.set_title(title, fontsize=13, loc="left")
        ax.grid(axis="y", color="0.9", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=11)
        ax.legend(fontsize=11, frameon=False, loc="lower left")
        ax.set_ylabel("Employment gap per 1 SD more exposure\n(%, relative to 2019; 95% CI shaded)", fontsize=11)
    axes[0].axvspan(2020.0, 2021.5, color="0.95", lw=0)
    axes[0].text(2020.75, axes[0].get_ylim()[1] * 0.9, "pandemic\n(dropped)", ha="center", fontsize=10, color=muted)
    for x, lab in [(2022.17, "first hike"), (2022.92, "ChatGPT")]:
        axes[0].axvline(x, color=muted, ls=":", lw=0.8)
        axes[0].text(x, axes[0].get_ylim()[0] * 0.95, f" {lab}", fontsize=10, color=muted, rotation=90, va="bottom")
    axes[1].axvline(2001.0, color=muted, ls=":", lw=0.8)
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=150)


def r4(o):
    """Round floats to 6 places, recursively. Four places caused double rounding in reported figures."""
    if isinstance(o, dict):
        return {k: r4(v) for k, v in o.items()}
    if isinstance(o, list):
        return [r4(v) for v in o]
    return round(float(o), 6) if isinstance(o, (float, np.floating)) else o


def main() -> None:
    rng = np.random.default_rng(2026)
    emp, t = load()
    out = {"spec": "docs/PREREGISTRATION.md, Test A"}

    print("ramp injection test (power first)...", flush=True)
    out["ramp_injection"] = ramp_injection(emp, t, rng)

    f, out["primary"], Y, x = run(emp, t, PRIMARY)
    c = es.contrast(f, PRIMARY.post_years, PRIMARY.base_year)
    out["primary"]["wild_bootstrap_p"] = es.wild_bootstrap_p(f, c, 999, rng)
    out["primary"]["randomization_p"] = randomization_p(Y, t, float(c @ f.beta), 1000, rng)
    adj = es.test(f, es.trend_adjusted(f, PRIMARY.pre_years, PRIMARY.ref_year, PRIMARY.post_years, PRIMARY.base_year),
                  es.quartile_gap(x))
    out["exploratory_trend_adjusted"] = {"delta_top_vs_bottom_quartile": adj,
                                         "prob_below_zero": es.prob_below_zero(adj["est"], adj["se"])}

    out["boom_controlled"] = run(emp, t, PRIMARY, {"B": boom(emp, list(Y.columns))})[1]
    out["dotcom"] = run(emp, t, DOTCOM)[1]
    out["primary_minus_dotcom_quartile"] = (out["primary"]["delta_top_vs_bottom_quartile"]["est"]
                                            - out["dotcom"]["delta_top_vs_bottom_quartile"]["est"])
    out["fake_2017"] = run(emp, t, FAKE)[1]

    # Shape of the dose-response: exposure quintiles (Q2..Q5 vs the least exposed fifth).
    quint = pd.qcut(x, 5, labels=False)
    fq = es.fit(Y, {f"Q{j + 1}": (quint == j).astype(float) for j in range(1, 5)} | controls(t, list(Y.columns)),
                PRIMARY.ref_year)
    out["quintiles_delta_vs_Q1"] = {v: es.test(fq, es.contrast(fq, PRIMARY.post_years, PRIMARY.base_year, v))
                                    for v in ["Q2", "Q3", "Q4", "Q5"]}

    out = r4(out)
    (RES / "event_study_industry.json").write_text(json.dumps(out, indent=2))
    plot(out["primary"], out["boom_controlled"], out["dotcom"], FIG / "fig_event_study_industry")
    for k in ["primary", "boom_controlled", "dotcom", "fake_2017"]:
        r = out[k]
        print(f"{k}: {r['n_industries']} industries, Delta (quartile) {r['delta_top_vs_bottom_quartile']}, "
              f"P(<0) {r['prob_below_zero']}, skeptical {r['prob_below_zero_skeptical']}, pre-trend p {r['pretrend']['p']}")
    print("wrote results/event_study_industry.json")


if __name__ == "__main__":
    main()
