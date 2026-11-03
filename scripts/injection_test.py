"""Injection tests: if AI had cut employment or hiring by X%, would our designs see it, and when?

Method (fixed before running, 2026-10-24). For each design:
  1. Draw a random set of CONTROL units and pretend they are treated. They had no AI-exposure
     shock, so any effect we find in them is noise.
  2. Plant a known effect in those fake-treated units: their outcome falls by `size` log points
     from 2023-Q1 onward (after ChatGPT), or not at all (size 0).
  3. Run the exact pipeline (synthetic DiD, primary 2019 baseline, same windows and
     deseasonalisation) against the remaining controls.
  4. Repeat 100 times per size. Report:
       - bias: average estimate minus the planted average effect over the post period;
       - power: share of draws where the estimate is below -1.96 x the null spread;
       - false alarm rate (size 0): share flagged anyway;
       - timing: share of draws where the path's first year below -size/2 is 2023 (correct),
         earlier (wrong: before the planted start), or never.
  Minimum detectable effect (MDE) = smallest planted size with power >= 80%.

Designs (primary outcome of each):
  ces      log CES employment, industries; fake treated = 22 of 44 controls
  qwi      young-minus-mid log hires, industry x region; fake treated = 6 of 12 control
           industries (all their regions together)
  cps      young-minus-old log employment, occupation groups; fake treated = 13 of 26 controls

Writes results/injection_test.json and results/injection_test.csv.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sdid import sdid  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
SIZES = [0.0, 0.025, 0.05, 0.10, 0.15]
REPS = 100
START = 2023.0


def ces_units() -> tuple[pd.DataFrame, list, float]:
    from sdid_industry import groups
    from synth_industry import load
    emp, t = load()
    c = groups(t, emp)["controls"]
    cols = emp.index[((emp.index >= 2003.0) & (emp.index <= 2019.75)) | ((emp.index >= 2022.0) & (emp.index <= 2026.5))]
    return emp.loc[cols, c].T, [[u] for u in c], 2019.75


def qwi_units() -> tuple[pd.DataFrame, list, float]:
    from build_industry_panel import industry_exposure
    from fetch_qwi import INDUSTRIES
    from sdid_qwi import load_qwi, panel
    codes = sorted(INDUSTRIES)
    exp = industry_exposure(pd.DataFrame({"id": codes, "prefix": codes})).set_index("id")
    dv, hu = exp["dv_rating_beta"], exp["human_rating_beta"]
    controls = sorted(exp.index[(dv < dv.median()) & (hu < hu.median())])
    tab = panel(load_qwi(), (2003.0, 2019.75), (2022.0, 2025.75), (2020.0, 2021.75))["gap"]
    tab = tab[tab.index.get_level_values("industry").isin(controls)]
    clusters = [list(tab.index[tab.index.get_level_values("industry") == c]) for c in controls]
    return tab, [cl for cl in clusters if cl], 2019.75


def cps_units() -> tuple[pd.DataFrame, list, float]:
    from sdid_occupation import groups, load, tables
    emp, g, n = load()
    c = groups(g, n)["controls"]
    tab = tables(emp, (2003.0, 2019.75), (2022.0, 2026.5), (2020.0, 2021.75))["gap"]
    tab = tab.loc[[u for u in c if u in tab.index]]
    return tab, [[u] for u in tab.index], 2019.75


def first_year_below(path: pd.Series, threshold: float) -> float | None:
    yr = path.groupby(np.floor(path.index)).mean()
    hit = yr[yr < threshold]
    return float(hit.index[0]) if len(hit) else None


def run_design(name: str, tab: pd.DataFrame, clusters: list, pre_end: float, rng) -> list[dict]:
    T_pre = int((tab.columns <= pre_end).sum())
    post_cols = tab.columns[T_pre:]
    share_after_start = float((post_cols >= START).mean())
    n_fake = len(clusters) // 2
    draws = [rng.permutation(len(clusters))[:n_fake] for _ in range(REPS)]
    null = []
    rows = []
    for size in SIZES:
        est, timing = [], []
        for d in draws:
            fake = [u for i in d for u in clusters[i]]
            rest = [u for u in tab.index if u not in fake]
            Yt = tab.loc[fake].to_numpy().copy()
            Yt[:, T_pre:][:, post_cols >= START] -= size
            r = sdid(tab.loc[rest].to_numpy(), Yt, T_pre)
            est.append(r.tau)
            if size > 0:
                timing.append(first_year_below(pd.Series(r.path[T_pre:], index=post_cols), -size / 2))
        est = np.array(est)
        if size == 0:
            null = est
        sd0 = float(np.std(null, ddof=1))
        planted = -size * share_after_start
        row = {"design": name, "size": size, "planted_avg_effect": round(planted, 4),
               "mean_estimate": round(float(est.mean()), 4), "bias": round(float(est.mean() - planted), 4),
               "null_sd": round(sd0, 4),
               "power": round(float(np.mean(est < np.mean(null) - 1.96 * sd0)), 2)}
        if size > 0:
            t = np.array([np.nan if x is None else x for x in timing])
            row["timing_correct_2023"] = round(float(np.mean(t == START)), 2)
            row["timing_too_early"] = round(float(np.mean(t < START)), 2)
            row["timing_never"] = round(float(np.mean(np.isnan(t))), 2)
        rows.append(row)
        print(row, flush=True)
    return rows


def main() -> None:
    rng = np.random.default_rng(2026)
    rows = []
    for name, loader in [("ces", ces_units), ("qwi", qwi_units), ("cps", cps_units)]:
        tab, clusters, pre_end = loader()
        print(f"\n{name}: {len(tab)} units, {len(clusters)} clusters, fake-treated clusters per draw {len(clusters) // 2}")
        rows += run_design(name, tab, clusters, pre_end, rng)
    df = pd.DataFrame(rows)
    mde = {d: (float(g.loc[g["power"] >= 0.8, "size"].min()) if (g["power"] >= 0.8).any() else None)
           for d, g in df[df["size"] > 0].groupby("design")}
    df.to_csv(RES / "injection_test.csv", index=False)
    (RES / "injection_test.json").write_text(json.dumps({"start": START, "reps": REPS, "sizes": SIZES,
                                                         "minimum_detectable_effect_80pct_power": mde,
                                                         "rows": rows}, indent=2))
    print("\nminimum detectable effect (80% power):", mde)


if __name__ == "__main__":
    main()
