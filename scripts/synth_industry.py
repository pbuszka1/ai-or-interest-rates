"""Industry-level synthetic control: computer systems design (NAICS 5415) vs low-exposure industries.

Second attempt after the CPS occupation version failed its hold-out check (project log,
2026-10-22). Outcome: log CES payroll employment, quarterly averages, 1990-Q1 to 2026-Q3.
Donors: industries below the median on BOTH AI exposure ratings, with complete data.

Decided before looking at results: the pandemic quarters 2020-Q1 to 2021-Q2 are left out of
the fit, because COVID hit the in-person, low-exposure donor industries far harder than tech
for reasons unrelated to rates or AI. A version that keeps them is reported as robustness.

Runs:
  1. Hold-out: fit 2003-2017, test 2018-Q1 to 2019-Q4 (the planned 2018-2021 test window,
     minus the pandemic).
  2. Main: fit 2003-2021 (pandemic excluded), gap 2022-Q1 to 2026-Q3.
  3. Robustness: main fit with the pandemic included.
  4. Dot-com: fit 1990-2000, gap 2001-2003 (CES has no 2003 coding break).
Placebo-in-space p-values for runs 2-4; balance on exposure, rate sensitivity, cyclicality.

Writes results/synth_industry.json and figures/fig_synth_industry_{main,dotcom}.{svg,png}.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synth import p_value, placebos, plot, synth  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROC, RES, FIG = ROOT / "data" / "processed", ROOT / "results", ROOT / "figures"
TREATED = "CES6054150001"
PANDEMIC = (2020.0, 2021.25)
EVENTS = [(2022.0, "First hike (Mar 2022)"), (2022.75, "ChatGPT (Nov 2022)")]


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    p = pd.read_parquet(PROC / "ces_panel.parquet")
    m = p.pivot(index="date", columns="id", values="value")
    q = m.groupby(m.index.year + (m.index.month - 1) // 3 / 4).mean()
    q = q[(q.index >= 1990) & (q.index <= 2026.5)]
    return np.log(q), pd.read_csv(PROC / "ces_industries.csv").set_index("id")


def donor_pool(emp: pd.DataFrame, t: pd.DataFrame) -> list[str]:
    low = (t["dv_rating_beta"] < t["dv_rating_beta"].median()) & \
          (t["human_rating_beta"] < t["human_rating_beta"].median())
    complete = emp.notna().all()
    return sorted(set(t.index[low]) & set(complete.index[complete]) - {TREATED})


def summarize(r, plc, t: pd.DataFrame, post_from: float) -> dict:
    ratio = r.rmspe_post / r.rmspe_fit
    post = r.gap[r.gap.index >= post_from]
    cols = ["dv_rating_beta", "human_rating_beta", "rate_sensitivity", "cyclicality"]
    w = r.weights[r.weights > 0.005].sort_values(ascending=False)
    syn = t.loc[r.weights.index, cols].mul(r.weights, axis=0).sum()
    return {
        "rmspe_fit": round(r.rmspe_fit, 4), "rmspe_post": round(r.rmspe_post, 4), "ratio": round(ratio, 2),
        "p_value": round(p_value(ratio, plc), 3), "n_placebos": len(plc),
        "gap_by_year": {int(y): round(float(v), 4) for y, v in post.groupby(np.floor(post.index)).mean().items()},
        "weights": {t.loc[k, "title"]: round(float(v), 3) for k, v in w.items()},
        "balance": {c: {"treated": round(float(t.loc[TREATED, c]), 3), "synthetic": round(float(syn[c]), 3)} for c in cols},
    }


def main() -> None:
    RES.mkdir(exist_ok=True)
    emp, t = load()
    donors = donor_pool(emp, t)
    y, X = emp[TREATED], emp[donors]
    out = {"treated": t.loc[TREATED, "title"], "n_donors": len(donors),
           "donors": [t.loc[d, "title"] for d in donors], "pandemic_excluded_from_fit": list(PANDEMIC)}
    print(f"donor pool: {len(donors)} industries")

    hold = synth(y, X, fit=(2003.0, 2017.75), post_end=2019.75)
    out["holdout"] = {"rmspe_fit_2003_17": round(hold.rmspe_fit, 4), "rmspe_2018_19": round(hold.rmspe_post, 4),
                      "ratio": round(hold.rmspe_post / hold.rmspe_fit, 2)}
    print("hold-out:", out["holdout"])

    main_r = synth(y, X, fit=(2003.0, 2021.75), exclude=PANDEMIC)
    plc = placebos(emp, donors, fit=(2003.0, 2021.75), exclude=PANDEMIC)
    out["main"] = summarize(main_r, plc, t, 2022.0)

    covid = synth(y, X, fit=(2003.0, 2021.75))
    plc_covid = placebos(emp, donors, fit=(2003.0, 2021.75))
    out["robust_pandemic_included"] = summarize(covid, plc_covid, t, 2022.0)

    dot = synth(y, X, fit=(1990.0, 2000.75), post_end=2003.75)
    plc_dot = placebos(emp, donors, fit=(1990.0, 2000.75), post_end=2003.75)
    out["dotcom"] = summarize(dot, plc_dot, t, 2001.0)

    # Is tech's hold-out degradation unusual? Same hold-out test with each donor as "treated".
    plc_hold = placebos(emp, donors, fit=(2003.0, 2017.75), post_end=2019.75)
    out["holdout"]["donor_ratio_median"] = round(float(plc_hold["ratio"].median()), 2)
    out["holdout"]["share_of_donors_with_worse_ratio"] = round(float((plc_hold["ratio"] >= out["holdout"]["ratio"]).mean()), 3)

    # Leave-one-out: drop each donor with weight above 5% and refit (Abadie's standard robustness check).
    out["leave_one_out"] = {}
    for d in main_r.weights[main_r.weights > 0.05].index:
        r = synth(y, X.drop(columns=d), fit=(2003.0, 2021.75), exclude=PANDEMIC)
        post = r.gap[r.gap.index >= 2022]
        out["leave_one_out"][f"without {t.loc[d, 'title']}"] = {
            "rmspe_fit": round(r.rmspe_fit, 4),
            "gap_by_year": {int(yr): round(float(v), 4) for yr, v in post.groupby(np.floor(post.index)).mean().items()},
            "top_weights": {t.loc[k, "title"]: round(float(v), 3) for k, v in r.weights.nlargest(3).items()}}

    (RES / "synth_industry.json").write_text(json.dumps(out, indent=2))
    plot(y, main_r, plc, (2003.0, 2021.75), EVENTS,
         "Computer systems design vs a synthetic blend of low-AI-exposure industries (CES payroll)",
         "Computer systems design", str(FIG / "fig_synth_industry_main"), start=2003.0)
    plot(y, dot, plc_dot, (1990.0, 2000.75), [(2001.0, "Bust begins")],
         "Dot-com check: same method, fit 1990–2000 (no generative AI existed)",
         "Computer systems design", str(FIG / "fig_synth_industry_dotcom"))

    for k in ["main", "robust_pandemic_included", "dotcom"]:
        s = out[k]
        print(f"\n{k}: RMSPE fit {s['rmspe_fit']}, post {s['rmspe_post']}, ratio {s['ratio']}, p = {s['p_value']}")
        print("  gap by year:", s["gap_by_year"])
        print("  weights:", s["weights"])
        print("  balance:", s["balance"])
    print("\nhold-out vs donors:", out["holdout"])
    for k, v in out["leave_one_out"].items():
        print(f"\n{k}: fit RMSPE {v['rmspe_fit']}, gap {v['gap_by_year']}, top weights {v['top_weights']}")


if __name__ == "__main__":
    main()
