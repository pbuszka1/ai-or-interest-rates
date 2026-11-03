"""Synthetic control for computer occupations (SOC 15-12): docs/METHODOLOGY.md, Step 3b.

Outcome: log employment, all ages 18-65, quarterly. Donors: occupation groups with
below-median AI exposure under BOTH the GPT-4 and human ratings, at least 150,000 workers in
2015-19 (smaller groups are too noisy quarter to quarter), and every quarter observed.

Three runs:
  1. Hold-out check: fit 2003-Q1 to 2017-Q4, then see how well the blend tracks 2018-2021.
  2. Main: fit 2003-Q1 to 2021-Q4, gap from 2022-Q1 to 2026-Q3.
  3. Dot-com: fit 1994-Q1 to 2000-Q4 (after the 1994 survey redesign), gap for 2001-2002
     (stopping before the 2003 coding break). No generative AI existed, so this shows what
     a plain tech bust looks like in this method.
Plus placebo-in-space for runs 2 and 3, and a balance table (exposure, cyclicality, boom).

Writes results/synth_computer.json, results/synth_computer_series.csv and two SVG figures.
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
TREATED = "15-12"
MIN_DONOR_SIZE = 150_000
CHATGPT, FIRST_HIKE = 2022.75, 2022.0  # 2022-Q4 (Nov 30, 2022); 2022-Q1 (Mar 2022)


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    p = pd.read_parquet(PROC / "panel_occ_age_quarter.parquet")
    emp = np.log(p.groupby(["t", "group"])["employed"].sum().unstack())
    g = pd.read_csv(PROC / "occ_groups.csv").set_index("group")
    rs = pd.read_csv(PROC / "occ_rate_sensitivity.csv").set_index("group")
    return emp, g.join(rs[["cyclicality", "rate_sensitivity_raw"]])


def donor_pool(emp: pd.DataFrame, g: pd.DataFrame) -> list[str]:
    low = (g["dv_rating_beta"] < g["dv_rating_beta"].median()) & \
          (g["human_rating_beta"] < g["human_rating_beta"].median())
    big = g["employed_2015_19"] >= MIN_DONOR_SIZE
    complete = emp.notna().all()
    return sorted(set(g.index[low & big]) & set(complete.index[complete]) - {TREATED})


def balance(weights: pd.Series, g: pd.DataFrame) -> dict:
    cols = ["dv_rating_beta", "human_rating_beta", "cyclicality", "rate_sensitivity_raw", "boom_2022"]
    syn = (g.loc[weights.index, cols].mul(weights, axis=0)).sum()
    return {c: {"computer": round(float(g.loc[TREATED, c]), 3), "synthetic": round(float(syn[c]), 3)} for c in cols}


def main() -> None:
    RES.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    emp, g = load()
    donors = donor_pool(emp, g)
    print(f"donor pool: {len(donors)} groups")
    out = {"treated": TREATED, "donors": donors}

    # 1. Hold-out: fit 2003-2017, check 2018-2021.
    hold = synth(emp[TREATED], emp[donors], fit=(2003.0, 2017.75), post_end=2021.75)
    out["holdout"] = {"rmspe_fit_2003_17": hold.rmspe_fit, "rmspe_2018_21": hold.rmspe_post,
                      "ratio": hold.rmspe_post / hold.rmspe_fit}
    print(f"hold-out: fit RMSPE {hold.rmspe_fit:.4f}, 2018-21 RMSPE {hold.rmspe_post:.4f} "
          f"(ratio {hold.rmspe_post / hold.rmspe_fit:.2f})")

    # 2. Main: fit 2003-2021, gap 2022-2026.
    main_r = synth(emp[TREATED], emp[donors], fit=(2003.0, 2021.75))
    plc = placebos(emp, donors, fit=(2003.0, 2021.75))
    ratio = main_r.rmspe_post / main_r.rmspe_fit
    post = main_r.gap[main_r.gap.index >= 2022]
    out["main"] = {
        "rmspe_fit": main_r.rmspe_fit, "rmspe_post": main_r.rmspe_post, "ratio": ratio,
        "p_value": p_value(ratio, plc), "n_placebos": len(plc),
        "weights": {k: round(v, 3) for k, v in main_r.weights[main_r.weights > 0.005].sort_values(ascending=False).items()},
        "gap_by_year": {int(y): round(float(v), 4) for y, v in post.groupby(np.floor(post.index)).mean().items()},
        "gap_2022q1": float(post.iloc[0]), "gap_latest": float(post.iloc[-1]),
        "balance": balance(main_r.weights, g),
    }

    # 3. Dot-com: fit 1994-2000, gap 2001-2002.
    dot = synth(emp[TREATED], emp[donors], fit=(1994.0, 2000.75), post_end=2002.75)
    plc_dot = placebos(emp, donors, fit=(1994.0, 2000.75), post_end=2002.75)
    dratio = dot.rmspe_post / dot.rmspe_fit
    dpost = dot.gap[dot.gap.index >= 2001]
    out["dotcom"] = {"rmspe_fit": dot.rmspe_fit, "rmspe_post": dot.rmspe_post, "ratio": dratio,
                     "p_value": p_value(dratio, plc_dot),
                     "gap_by_year": {int(y): round(float(v), 4) for y, v in dpost.groupby(np.floor(dpost.index)).mean().items()},
                     "weights": {k: round(v, 3) for k, v in dot.weights[dot.weights > 0.005].sort_values(ascending=False).items()}}

    (RES / "synth_computer.json").write_text(json.dumps(out, indent=2, default=float))
    pd.DataFrame({"computer": emp.loc[main_r.synthetic.index, TREATED], "synthetic": main_r.synthetic,
                  "gap": main_r.gap}).to_csv(RES / "synth_computer_series.csv", index_label="t")

    # Main plot starts in 2003 (earlier quarters use the old occupation codes); the dot-com plot
    # starts in 1994 (the survey redesign).
    plot(emp[TREATED], main_r, plc, (2003.0, 2021.75), [(FIRST_HIKE, "First hike"), (CHATGPT, "ChatGPT")],
         "Computer occupations vs a synthetic blend of low-AI-exposure occupations (fit 2003–2021)",
         "Computer occupations", str(FIG / "fig_synth_computer_main"), start=2003.0)
    plot(emp[TREATED], dot, plc_dot, (1994.0, 2000.75), [(2001.0, "Bust begins")],
         "Dot-com check: same method, fit 1994–2000 (no generative AI existed)",
         "Computer occupations", str(FIG / "fig_synth_computer_dotcom"), start=1994.0)

    m, d = out["main"], out["dotcom"]
    print(f"main: fit RMSPE {m['rmspe_fit']:.4f}, post RMSPE {m['rmspe_post']:.4f}, ratio {m['ratio']:.2f}, "
          f"permutation p = {m['p_value']:.2f} ({m['n_placebos']} placebos)")
    print("main gap by year (log points):", m["gap_by_year"])
    print("weights:", m["weights"])
    print("balance:", json.dumps(m["balance"]))
    print(f"dot-com: fit RMSPE {d['rmspe_fit']:.4f}, post RMSPE {d['rmspe_post']:.4f}, ratio {d['ratio']:.2f}, "
          f"p = {d['p_value']:.2f}; gap by year {d['gap_by_year']}")
    print("dot-com weights:", d["weights"])


if __name__ == "__main__":
    main()
