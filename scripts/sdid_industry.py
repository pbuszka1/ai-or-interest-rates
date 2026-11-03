"""Pooled synthetic DiD on CES industries: high- vs low-AI-exposure, 2022 and the dot-com bust.

Design fixed before running (2026-10-23):
  Outcome   log CES payroll employment, quarterly averages.
  Controls  industries below the median on BOTH exposure ratings (GPT-4 beta, human beta).
  Treated   "high": top quarter (>= 75th percentile) on BOTH ratings.
            "middle": between the median and the 75th percentile on both (dose-response).
            "computer systems design" alone (comparable with the synthetic control runs).
  Windows   2022: pre 2003-Q1 to 2021-Q4 with the pandemic quarters (2020-Q1 to 2021-Q2)
            dropped; post 2022-Q1 to 2026-Q3. 2022-Q1 is the boundary so a break at the rate
            hikes would be visible; ChatGPT (2022-Q4) falls inside the post period.
            Dot-com: pre 1990-Q1 to 2000-Q4, post 2001-Q1 to 2003-Q4. No generative AI
            existed, so this measures how much high-exposure industries fall relative to
            low-exposure ones in a tech bust without AI.
            Placebo in time: pre 2003-Q1 to 2015-Q4, post 2016-Q1 to 2019-Q4 (no shock;
            the estimate should be near zero).
  Inference placebo standard errors, 200 draws.
  Balance   treated mean vs weighted controls on rate sensitivity and cyclicality (CES, 1990-2019).

Writes results/sdid_industry.json and figures/fig_sdid_industry.{svg,png}.
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
from sdid import did, placebo_se, sdid, summarize  # noqa: E402
from synth_industry import PANDEMIC, TREATED, load  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES, FIG = ROOT / "results", ROOT / "figures"


def groups(t: pd.DataFrame, emp: pd.DataFrame) -> dict[str, list[str]]:
    complete = set(emp.columns[emp.notna().all()])
    dv, hu = t["dv_rating_beta"], t["human_rating_beta"]
    def both(lo, hi):
        """Industries in the [lo, hi) quantile band on both ratings (hi = 1 includes the maximum)."""
        def band(s):
            return (s >= s.quantile(lo)) & ((s < s.quantile(hi)) if hi < 1 else True)
        return sorted(set(t.index[band(dv) & band(hu)]) & complete)
    return {"controls": both(0.0, 0.5), "high": both(0.75, 1.0), "middle": both(0.5, 0.75),
            "computer systems design": [TREATED]}


def window(emp: pd.DataFrame, pre: tuple[float, float], post: tuple[float, float],
           drop: tuple[float, float] | None = None) -> tuple[pd.DataFrame, int]:
    cols = emp.index[(emp.index >= pre[0]) & (emp.index <= post[1])]
    if drop:
        cols = cols[~((cols >= drop[0]) & (cols <= drop[1]))]
    return emp.loc[cols], int((cols <= pre[1]).sum())


def run(emp, t, treated, controls, pre, post, drop=None, se=True) -> dict:
    w, T_pre = window(emp, pre, post, drop)
    Yc, Yt = w[controls].T.to_numpy(), w[treated].T.to_numpy()
    r = sdid(Yc, Yt, T_pre)
    out = {"n_treated": len(treated), "n_controls": len(controls), "tau": round(r.tau, 6),
           "did_unweighted": round(did(Yc, Yt, T_pre), 6), "max_control_weight": round(float(r.omega.max()), 2),
           **summarize(r, w.index, T_pre)}
    if se and len(controls) > len(treated):
        out["se"] = round(placebo_se(Yc, len(treated), T_pre, reps=200), 6)
        out["ci95"] = [round(out["tau"] - 1.96 * out["se"], 6), round(out["tau"] + 1.96 * out["se"], 6)]
    cols = ["rate_sensitivity", "cyclicality"]
    out["balance"] = {c: {"treated": round(float(t.loc[treated, c].mean()), 3),
                          "weighted_controls": round(float(t.loc[controls, c].to_numpy() @ r.omega), 3)} for c in cols}
    return out


def _path(r: dict) -> pd.Series:
    p = pd.Series(r["path"]).rename(index=float).sort_index()
    # Break the line across dropped quarters (the pandemic) instead of drawing a straight bridge.
    gaps = p.index.to_series().diff() > 0.3
    return pd.concat([p, pd.Series(np.nan, index=p.index[gaps] - 0.01)]).sort_index()


def plot(res: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    panels = [(axes[0], [("2022_high", "Baseline: late 2021 (main)", "#1f4e79"),
                         ("2022_high_pre_ends_2019", "Baseline: 2019 (pre-pandemic)", "#c0504d")],
               "2022: high vs low AI exposure", [(2022.0, "First hike"), (2022.75, "ChatGPT")]),
              (axes[1], [("dotcom_high", "Baseline: late 2000", "#1f4e79")],
               "Dot-com: same industries, no generative AI", [(2001.0, "Bust")])]
    for ax, lines, title, mark in panels:
        se = res[lines[0][0]].get("se", np.nan)
        ax.axhspan(-196 * se, 196 * se, color="0.92", label="±1.96 placebo s.e. (main)")
        for key, lab, col in lines:
            p = _path(res[key])
            ax.plot(p.index, 100 * p.values, color=col, lw=1.8, marker="o", ms=2,
                    label=f"{lab}: τ = {100 * res[key]['tau']:+.1f}% (s.e. {100 * res[key].get('se', np.nan):.1f})")
        ax.axhline(0, color="0.3", lw=0.8)
        if ax is axes[0]:
            ax.axvspan(2020.0, 2021.5, color="0.97", hatch="///", lw=0, label="Pandemic quarters (dropped)")
        lo = ax.get_ylim()[0]
        for i, (x, lab) in enumerate(mark):
            ax.axvline(x, color="0.4", ls=":", lw=0.8)
            ax.text(x, lo + 0.6 + 0.8 * i, f" {lab}", fontsize=8, color="0.3", va="bottom")
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("Quarter")
        ax.legend(frameon=False, loc="upper left", fontsize=7.5)
    axes[0].set_ylabel("Effect on log employment (×100)\nhigh-exposure minus weighted low-exposure")
    fig.tight_layout()
    fig.savefig(FIG / "fig_sdid_industry.svg")
    fig.savefig(FIG / "fig_sdid_industry.png", dpi=120)
    plt.close(fig)


def main() -> None:
    emp, t = load()
    g = groups(t, emp)
    c = g["controls"]
    print({k: len(v) for k, v in g.items()})
    print("high-exposure industries:", [t.loc[i, "title"] for i in g["high"]])
    res = {"groups": {k: [t.loc[i, "title"] for i in v] for k, v in g.items()}}
    W22, DOT, PLT = ((2003.0, 2021.75), (2022.0, 2026.5)), ((1990.0, 2000.75), (2001.0, 2003.75)), ((2003.0, 2015.75), (2016.0, 2019.75))

    res["2022_high"] = run(emp, t, g["high"], c, *W22, drop=PANDEMIC)
    res["2022_middle"] = run(emp, t, g["middle"], c, *W22, drop=PANDEMIC)
    res["2022_computer_systems_design"] = run(emp, t, g["computer systems design"], c, *W22, drop=PANDEMIC)
    res["2022_high_pandemic_kept"] = run(emp, t, g["high"], c, *W22)
    # Added after the main run put 82% of its time weight on 2021-Q4 (post-pandemic boom):
    # same comparison with a pre-pandemic baseline, 2020-2021 dropped entirely.
    res["2022_high_pre_ends_2019"] = run(emp, t, g["high"], c, (2003.0, 2019.75), (2022.0, 2026.5), drop=(2020.0, 2021.75))
    res["2022_middle_pre_ends_2019"] = run(emp, t, g["middle"], c, (2003.0, 2019.75), (2022.0, 2026.5), drop=(2020.0, 2021.75))
    res["dotcom_high"] = run(emp, t, g["high"], c, *DOT)
    res["dotcom_computer_systems_design"] = run(emp, t, g["computer systems design"], c, *DOT)
    res["placebo_2016_high"] = run(emp, t, g["high"], c, *PLT)

    (RES / "sdid_industry.json").write_text(json.dumps(res, indent=2))
    plot(res)
    for k, v in res.items():
        if k == "groups":
            continue
        print(f"\n{k}: tau {v['tau']:+.4f}  se {v.get('se', float('nan')):.4f}  ci {v.get('ci95')}  "
              f"(unweighted DiD {v['did_unweighted']:+.4f}; pre-path rms {v['pre_path_rms']}; "
              f"effective controls {v['effective_controls']}, max weight {v['max_control_weight']})")
        print("   by year:", v["by_year"])
        print("   balance:", v["balance"])


if __name__ == "__main__":
    main()
