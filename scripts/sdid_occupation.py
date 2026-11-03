"""Synthetic DiD at occupation level (CPS): the closest public-data match to the Stanford design.

Design fixed before running (2026-10-23):
  Units     SOC minor occupation groups from data/processed/panel_occ_age_quarter.parquet.
            Kept only if they average at least MIN_YOUNG employed CPS respondents aged 22-27
            per quarter over 2015-2019 and have employment in every quarter of each window
            (thin cells make log employment jump around).
  Exposure  group-level GPT-4 and human beta ratings (occ_groups.csv). Treated "high" = top
            quarter on BOTH; controls = below the median on both; "middle" = between the median
            and 75th percentile on both; "computer occupations" (15-12) alone.
  Outcomes  young = log employment ages 22-27; old = log employment ages 35-65;
            gap = young - old in the same occupation (the key outcome: anything hitting the
            whole occupation cancels).
  Windows   primary   pre 2003-Q1..2019-Q4, post 2022-Q1..2026-Q3, 2020-2021 dropped
            secondary pre 2003-Q1..2021-Q4 minus 2020-Q1..2021-Q2, same post
            exploratory pre 2003-Q1..2019-Q4, post 2020-Q1..2026-Q3 (shows when a gap opens)
            dot-com   pre 1994-Q1..2000-Q4 (after the 1994 survey redesign), post
                      2001-Q1..2002-Q4 (stops before the 2003 occupation-code break)
  Seasonality each unit deseasonalised by its pre-period quarter-of-year means (rule since
            the QWI run).
  Inference placebo standard errors (200 draws) when controls outnumber treated units,
            otherwise the jackknife; the method used is recorded with each estimate.

Writes results/sdid_occupation.json and figures/fig_sdid_occupation.{svg,png}.
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sdid import jackknife_se, placebo_se, sdid, summarize  # noqa: E402
from sdid_qwi import deseason  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROC, RES, FIG = ROOT / "data" / "processed", ROOT / "results", ROOT / "figures"
MIN_YOUNG = 60
YOUNG, OLD = "22-27", "35-65"
WINDOWS = {
    "primary_2019_baseline": ((2003.0, 2019.75), (2022.0, 2026.5), (2020.0, 2021.75)),
    "secondary_late2021_baseline": ((2003.0, 2021.75), (2022.0, 2026.5), (2020.0, 2021.25)),
    "exploratory_pandemic_kept": ((2003.0, 2019.75), (2020.0, 2026.5), None),
    "dotcom": ((1994.0, 2000.75), (2001.0, 2002.75), None),
}


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p = pd.read_parquet(PROC / "panel_occ_age_quarter.parquet")
    g = pd.read_csv(PROC / "occ_groups.csv").set_index("group")
    emp = p.pivot_table(index=["group", "age_group"], columns="t", values="employed", observed=True)
    n = p[(p["year"] >= 2015) & (p["year"] <= 2019) & (p["age_group"] == YOUNG)] \
        .groupby("group")["n_respondents_employed"].mean()
    return emp, g, n


def groups(g: pd.DataFrame, n: pd.Series) -> dict[str, list[str]]:
    big = set(n.index[n >= MIN_YOUNG])
    dv, hu = g["dv_rating_beta"], g["human_rating_beta"]
    def band(s, lo, hi):
        return (s >= s.quantile(lo)) & ((s < s.quantile(hi)) if hi < 1 else True)
    pick = lambda lo, hi: sorted(set(g.index[band(dv, lo, hi) & band(hu, lo, hi)]) & big)
    return {"controls": pick(0.0, 0.5), "high": pick(0.75, 1.0), "middle": pick(0.5, 0.75),
            "computer occupations": ["15-12"]}


def tables(emp: pd.DataFrame, pre, post, drop) -> dict[str, pd.DataFrame]:
    cols = emp.columns[(emp.columns >= pre[0]) & (emp.columns <= post[1])]
    if drop:
        cols = cols[~((cols >= drop[0]) & (cols <= drop[1]))]
    e = np.log(emp[cols].where(emp[cols] > 0))
    young, old = e.xs(YOUNG, level="age_group"), e.xs(OLD, level="age_group")
    ok = young.notna().all(axis=1) & old.reindex(young.index).notna().all(axis=1)
    young, old = young[ok], old.loc[young.index[ok]]
    return {k: deseason(v, pre[1]) for k, v in {"gap": young - old, "young": young, "old": old}.items()}


def run(tab: pd.DataFrame, treated: list[str], controls: list[str], pre_end: float) -> dict:
    tr, co = [u for u in treated if u in tab.index], [u for u in controls if u in tab.index]
    Yt, Yc = tab.loc[tr].to_numpy(), tab.loc[co].to_numpy()
    T_pre = int((tab.columns <= pre_end).sum())
    r = sdid(Yc, Yt, T_pre)
    # Placebo inference needs more controls than treated units; otherwise fall back to the jackknife.
    if len(co) > len(tr):
        se, method = placebo_se(Yc, len(tr), T_pre, reps=200), "placebo"
    else:
        se, method = jackknife_se(Yc, Yt, T_pre), "jackknife"
    return {"n_treated": len(tr), "n_controls": len(co), "tau": round(r.tau, 6), "se": round(se, 6),
            "se_method": method, "ci95": [round(r.tau - 1.96 * se, 6), round(r.tau + 1.96 * se, 6)],
            **summarize(r, tab.columns, T_pre)}


def plot(res: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, key, title, marks in [
            (axes[0], "exploratory_pandemic_kept", "Occupations: young-minus-old employment gap, high vs low exposure\n(2019 baseline, pandemic quarters shown; exploratory)",
             [(2020.0, "Pandemic"), (2022.75, "ChatGPT")]),
            (axes[1], "dotcom", "Dot-com: same comparison, no generative AI", [(2001.0, "Bust")])]:
        r = res[key]["high"]["gap"]
        p = pd.Series(r["path"]).rename(index=float).sort_index()
        ax.axhspan(-196 * r["se"], 196 * r["se"], color="0.92", label=f"±1.96 {r['se_method']} s.e.")
        ax.plot(p.index, 100 * p.values, color="#1f4e79", lw=1.4, marker="o", ms=2,
                label=f"τ = {100 * r['tau']:+.1f}% (s.e. {100 * r['se']:.1f})")
        ax.axhline(0, color="0.3", lw=0.8)
        hi = ax.get_ylim()[1]
        for i, (x, lab) in enumerate(marks):
            ax.axvline(x, color="0.4", ls=":", lw=0.8)
            ax.text(x, hi - 3 * i, f" {lab}", fontsize=8, color="0.3", va="top")
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("Quarter")
        ax.legend(frameon=False, loc="lower left", fontsize=8)
    axes[0].set_ylabel("Effect on log(employed 22–27) − log(employed 35–65), ×100")
    fig.tight_layout()
    fig.savefig(FIG / "fig_sdid_occupation.svg")
    fig.savefig(FIG / "fig_sdid_occupation.png", dpi=120)
    plt.close(fig)


def main() -> None:
    emp, g, n = load()
    grp = groups(g, n)
    title = lambda c: f"{c} (largest: OCC2010 {int(g.loc[c, 'largest_occ2010'])})"
    res = {"min_young_respondents_per_quarter": MIN_YOUNG,
           "groups": {k: [title(c) for c in v] for k, v in grp.items()}}
    print({k: len(v) for k, v in grp.items()})
    for name, (pre, post, drop) in WINDOWS.items():
        tabs = tables(emp, pre, post, drop)
        res[name] = {}
        for treat in ["high", "middle", "computer occupations"]:
            res[name][treat] = {o: run(tabs[o], grp[treat], grp["controls"], pre[1]) for o in ["gap", "young", "old"]}
        print(f"\n{name}:")
        for treat in ["high", "middle", "computer occupations"]:
            for o in ["gap", "young", "old"]:
                r = res[name][treat][o]
                print(f"  {treat:21s} {o:5s} tau {r['tau']:+.4f} se {r['se']:.4f} ({r['se_method']}) ci {r['ci95']} "
                      f"n {r['n_treated']}/{r['n_controls']} pre-rms {r['pre_path_rms']} by year {r['by_year']}")
    (RES / "sdid_occupation.json").write_text(json.dumps(res, indent=2))
    plot(res)


if __name__ == "__main__":
    main()
