"""Synthetic DiD on QWI hires by age: did hiring of young workers fall more in AI-exposed industries?

QWI counts actual hires from state unemployment-insurance records, by industry and age band.
This is where an early-career AI effect (the Stanford "canaries" finding) should show up.

Design fixed before running (2026-10-23):
  Units     industry (27 NAICS 4-digit industries from scripts/fetch_qwi.py) x Census region
            (4) = 108 units. Hires are summed over a FIXED set of states per window: states
            reporting in every quarter of the window, so entry and exit can't create changes.
  Exposure  each industry's employment-weighted occupational AI exposure (CPS 2020-2025),
            same method as the CES industries. Treated = above the median on BOTH ratings;
            controls = below the median on both; the rest left out.
  Outcomes  young     log hires, ages 22-24 (QWI agegrp A03)
            gap       log hires 22-24 minus log hires 35-44 (A05), same industry and region.
                      Anything hitting an industry's hiring as a whole (e.g. a rate shock)
                      cancels; what's left is age-specific. This is the key outcome.
            mid       log hires, ages 35-44 (reference)
  Windows   primary   pre 2003-Q1 to 2019-Q4, post 2022-Q1 to 2025-Q4 (2020-2021 dropped):
                      pre-pandemic baseline, the project's rule since 2026-10-23.
            secondary pre 2003-Q1 to 2021-Q4 minus 2020-Q1 to 2021-Q2, same post.
            dot-com   pre 1996-Q1 to 2000-Q4, post 2001-Q1 to 2003-Q4 (states reporting from 1996).
  Inference placebo standard errors, 200 draws; time weights reported.

Changes after the first run (2026-10-23, logged):
  - Placebo inference needs more control than treated units; the median split gave 48 vs 47,
    so every s.e. was NaN. Inference is now a jackknife clustered by industry (each industry's
    4 regional units leave together), the paper's alternative when treated units are many.
  - The quarterly paths swung about +/-15 points with the seasons (young hiring is seasonal,
    differently by industry, even in QWI's "sa" series), and time weights could lock onto one
    season (45% on 2019-Q2). Each unit's outcomes are now deseasonalised: subtract that unit's
    pre-period mean for each quarter of the year.
  - EXPLORATORY diagnostic added after seeing that the primary gap estimate was already
    -15% in 2022-Q1: same comparison with the pandemic quarters kept (pre 2003-2019, post
    2020-Q1 to 2025-Q4) to see *when* the gap opened.

Writes results/sdid_qwi.json and figures/fig_sdid_qwi.{svg,png}.
"""

import glob
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_industry_panel import industry_exposure  # noqa: E402
from fetch_qwi import INDUSTRIES  # noqa: E402
from sdid import jackknife_se, sdid, summarize  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES, FIG = ROOT / "results", ROOT / "figures"
REGION = {  # Census regions by state FIPS: 1 Northeast, 2 Midwest, 3 South, 4 West
    **dict.fromkeys([9, 23, 25, 33, 44, 50, 34, 36, 42], 1),
    **dict.fromkeys([17, 18, 26, 39, 55, 19, 20, 27, 29, 31, 38, 46], 2),
    **dict.fromkeys([10, 11, 12, 13, 24, 37, 45, 51, 54, 1, 21, 28, 47, 5, 22, 40, 48], 3),
    **dict.fromkeys([4, 8, 16, 30, 32, 35, 49, 56, 2, 6, 15, 41, 53], 4),
}
WINDOWS = {
    "primary_2019_baseline": ((2003.0, 2019.75), (2022.0, 2025.75), (2020.0, 2021.75)),
    "secondary_late2021_baseline": ((2003.0, 2021.75), (2022.0, 2025.75), (2020.0, 2021.25)),
    "dotcom": ((1996.0, 2000.75), (2001.0, 2003.75), None),
    "exploratory_pandemic_kept": ((2003.0, 2019.75), (2020.0, 2025.75), None),
}


def load_qwi() -> pd.DataFrame:
    frames = [pd.read_csv(f, dtype={"industry": str, "state": int, "agegrp": str})
              for f in glob.glob(str(ROOT / "data" / "raw" / "qwi" / "qwi_*.csv"))]
    q = pd.concat([f for f in frames if len(f)], ignore_index=True)
    q["t"] = q["time"].str[:4].astype(int) + (q["time"].str[-1].astype(int) - 1) / 4
    q["HirA"] = pd.to_numeric(q["HirA"], errors="coerce")
    q["region"] = q["state"].map(REGION)
    return q[q["agegrp"].isin(["A03", "A05"])]


def panel(q: pd.DataFrame, pre: tuple, post: tuple, drop: tuple | None) -> dict[str, pd.DataFrame]:
    """Unit x quarter tables of log hires for each outcome, over states reporting throughout."""
    span = q[(q["t"] >= pre[0]) & (q["t"] <= post[1])]
    need = span["t"].nunique()
    per_state = span[span["agegrp"] == "A03"].groupby("state")["t"].nunique()
    states = per_state.index[per_state == need]
    s = span[span["state"].isin(states)]
    if drop:
        s = s[~((s["t"] >= drop[0]) & (s["t"] <= drop[1]))]
    h = s.groupby(["industry", "region", "agegrp", "t"])["HirA"].sum(min_count=1).unstack("t")
    h = h.where(h > 0)  # zero hires in a cell can't be logged; those units are dropped below
    young, mid = np.log(h.xs("A03", level="agegrp")), np.log(h.xs("A05", level="agegrp"))
    ok = young.notna().all(axis=1) & mid.notna().all(axis=1)
    out = {"young": young[ok], "gap": (young - mid)[ok], "mid": mid[ok]}
    out = {k: deseason(v, pre[1]) for k, v in out.items()}
    out["n_states"] = len(states)
    return out


def deseason(tab: pd.DataFrame, pre_end: float) -> pd.DataFrame:
    """Remove each unit's own seasonal pattern: subtract its pre-period mean by quarter of year."""
    qoy = np.round((tab.columns % 1) * 4).astype(int)
    pre = tab.columns <= pre_end
    out = tab.copy()
    for k in range(4):
        cols = qoy == k
        out.loc[:, cols] = tab.loc[:, cols].sub(tab.loc[:, cols & pre].mean(axis=1), axis=0)
    return out


def run(tab: pd.DataFrame, treated_ind: list[str], control_ind: list[str], pre_end: float) -> dict:
    ind = tab.index.get_level_values("industry")
    is_t, is_c = ind.isin(treated_ind), ind.isin(control_ind)
    Yt, Yc = tab[is_t].to_numpy(), tab[is_c].to_numpy()
    T_pre = int((tab.columns <= pre_end).sum())
    r = sdid(Yc, Yt, T_pre)
    # Clustered by industry: an industry's four regional units leave the jackknife together.
    se = jackknife_se(Yc, Yt, T_pre, cl_co=np.asarray(ind[is_c]), cl_tr=np.asarray(ind[is_t]))
    return {"n_treated_units": int(Yt.shape[0]), "n_control_units": int(Yc.shape[0]),
            "tau": round(r.tau, 6), "se": round(se, 6), "ci95": [round(r.tau - 1.96 * se, 6), round(r.tau + 1.96 * se, 6)],
            **summarize(r, tab.columns, T_pre)}


def plot(res: dict) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 9), sharey=True)   # stacked, so each panel prints full width
    for ax, key, title in [(axes[0], "exploratory_pandemic_kept",
                            "Young-minus-mid hiring gap, high vs low exposure\n(2019 baseline, pandemic quarters shown; exploratory)"),
                           (axes[1], "dotcom", "Dot-com: same comparison, no generative AI")]:
        r = res[key]["gap"]
        p = pd.Series(r["path"]).rename(index=float).sort_index()
        gaps = p.index.to_series().diff() > 0.3
        p = pd.concat([p, pd.Series(np.nan, index=p.index[gaps] - 0.01)]).sort_index()
        ax.axhspan(-196 * r["se"], 196 * r["se"], color="0.92", label="±1.96 jackknife s.e. (by industry)")
        ax.plot(p.index, 100 * p.values, color="#1f4e79", lw=1.6, marker="o", ms=2,
                label=f"τ = {100 * r['tau']:+.1f}% (s.e. {100 * r['se']:.1f})")
        ax.axhline(0, color="0.3", lw=0.8)
        hi = ax.get_ylim()[1]
        for i, (x, lab) in enumerate([(2020.0, "Pandemic"), (2022.75, "ChatGPT")] if ax is axes[0] else [(2001.0, "Bust")]):
            ax.axvline(x, color="0.4", ls=":", lw=0.8)
            ax.text(x, hi - 3 * i, f" {lab}", fontsize=11, color="0.3", va="top")
        ax.set_title(title, loc="left", fontsize=13)
        ax.set_xlabel("Quarter", fontsize=11)
        ax.tick_params(labelsize=11)
        ax.legend(frameon=False, loc="lower left", fontsize=11)
        ax.set_ylabel("Effect on log(hires 22–24)\n− log(hires 35–44), ×100", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig_sdid_qwi.svg")
    fig.savefig(FIG / "fig_sdid_qwi.png", dpi=150)
    plt.close(fig)


def main() -> None:
    q = load_qwi()
    codes = sorted(INDUSTRIES)
    exp = industry_exposure(pd.DataFrame({"id": codes, "prefix": codes})).set_index("id")
    dv, hu = exp["dv_rating_beta"], exp["human_rating_beta"]
    treated = sorted(exp.index[(dv > dv.median()) & (hu > hu.median())])
    controls = sorted(exp.index[(dv < dv.median()) & (hu < hu.median())])
    res = {"treated_industries": {c: INDUSTRIES[c] for c in treated},
           "control_industries": {c: INDUSTRIES[c] for c in controls},
           "exposure": exp[["dv_rating_beta", "human_rating_beta"]].round(3).to_dict(orient="index")}
    print("treated:", [INDUSTRIES[c] for c in treated])
    print("controls:", [INDUSTRIES[c] for c in controls])

    for name, (pre, post, drop) in WINDOWS.items():
        tabs = panel(q, pre, post, drop)
        res[name] = {"n_states": int(tabs["n_states"])}
        for outcome in ["gap", "young", "mid"]:
            res[name][outcome] = run(tabs[outcome], treated, controls, pre[1])
        print(f"\n{name} ({tabs['n_states']} states):")
        for outcome in ["gap", "young", "mid"]:
            r = res[name][outcome]
            print(f"  {outcome:5s} tau {r['tau']:+.4f} se {r['se']:.4f} ci {r['ci95']} "
                  f"units {r['n_treated_units']}/{r['n_control_units']} pre-rms {r['pre_path_rms']} by year {r['by_year']}")
            print(f"        time weights {r['time_weights_over_1pct']}")

    (RES / "sdid_qwi.json").write_text(json.dumps(res, indent=2))
    plot(res)


if __name__ == "__main__":
    main()
