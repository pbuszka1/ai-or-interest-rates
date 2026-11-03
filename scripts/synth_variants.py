"""Can the industry synthetic control be made more robust? Compare standard variants side by side.

Selection rule, written BEFORE running (2026-10-23): among variants whose blend spreads
weight over at least 4 effective donors (1 / sum of squared weights), pick the one with the
lowest hold-out error (fit 2003-2017, test 2018-2019). The size of the post-2022 gap plays
no part in the choice. Every variant is reported regardless.

Variants:
  A. baseline           classic demeaned SC on log levels (the 2026-10-23 run)
  B. ridge              same, plus a penalty that spreads weight across donors; penalty
                        strength chosen on the hold-out, never on post-2022 data
  C. growth             match year-over-year growth instead of levels, so a donor doesn't
                        need tech's long-run trend, only its ups and downs (gap cumulated
                        back into levels for comparison)
  D. wider pool         donors below the 75th exposure percentile on both ratings (more and
                        more similar donors, at the cost of some AI contamination, which
                        biases the gap toward zero)
  E. wider pool + ridge D and B together

Writes results/synth_variants.json and results/synth_variants.csv.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synth import p_value, placebos, synth  # noqa: E402
from synth_industry import PANDEMIC, TREATED, load  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIT, HOLD = (2003.0, 2021.75), (2003.0, 2017.75)
RIDGE_GRID = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
MIN_EFFECTIVE_DONORS = 4


def pool(emp: pd.DataFrame, t: pd.DataFrame, q: float) -> list[str]:
    low = (t["dv_rating_beta"] < t["dv_rating_beta"].quantile(q)) & \
          (t["human_rating_beta"] < t["human_rating_beta"].quantile(q))
    complete = emp.notna().all()
    return sorted(set(t.index[low]) & set(complete.index[complete]) - {TREATED})


def pick_ridge(y: pd.Series, X: pd.DataFrame) -> float:
    """Ridge strength with the lowest hold-out error (2018-19), using pre-2020 data only."""
    errs = {r: synth(y, X, fit=HOLD, post_end=2019.75, ridge=r).rmspe_post for r in RIDGE_GRID}
    return min(errs, key=errs.get)


def evaluate(name: str, y: pd.Series, X: pd.DataFrame, emp: pd.DataFrame, donors: list[str],
             t: pd.DataFrame, ridge: float = 0.0, growth: bool = False) -> dict:
    hold = synth(y, X, fit=HOLD, post_end=2019.75, ridge=ridge)
    r = synth(y, X, fit=FIT, exclude=PANDEMIC, ridge=ridge)
    plc = placebos(emp, donors, fit=FIT, exclude=PANDEMIC, ridge=ridge)
    gap = r.gap
    if growth:  # a year-over-year gap / 4 ~ a quarterly growth gap; summing those from 2022-Q1 gives a level gap
        gap = gap.where(gap.index >= 2022, 0.0).cumsum() / 4
    post = gap[gap.index >= 2022]
    w = r.weights
    # Leave out the top donor and see whether the 2022 gap keeps its sign and rough size.
    top = w.idxmax()
    lo = synth(y, X.drop(columns=top), fit=FIT, exclude=PANDEMIC, ridge=ridge).gap
    if growth:
        lo = lo.where(lo.index >= 2022, 0.0).cumsum() / 4
    lo_post = lo[lo.index >= 2022]
    cols = ["dv_rating_beta", "rate_sensitivity", "cyclicality"]
    syn = t.loc[w.index, cols].mul(w, axis=0).sum()
    ratio = r.rmspe_post / r.rmspe_fit
    return {
        "variant": name, "n_donors": len(donors), "ridge": ridge,
        "fit_error": round(r.rmspe_fit, 4), "holdout_error_2018_19": round(hold.rmspe_post, 4),
        "effective_donors": round(float(1 / np.sum(w ** 2)), 1), "max_weight": round(float(w.max()), 2),
        "top_donor": t.loc[top, "title"],
        "cyclicality_tech_vs_synth": f"{t.loc[TREATED, 'cyclicality']:.2f} vs {syn['cyclicality']:.2f}",
        "gap_2022": round(float(post[post.index < 2023].mean()), 3),
        "gap_2024": round(float(post[(post.index >= 2024) & (post.index < 2025)].mean()), 3),
        "gap_2026": round(float(post[post.index >= 2026].mean()), 3),
        "p_value": round(p_value(ratio, plc), 2),
        "gap_2022_without_top_donor": round(float(lo_post[lo_post.index < 2023].mean()), 3),
        "gap_2026_without_top_donor": round(float(lo_post[lo_post.index >= 2026].mean()), 3),
    }


def main() -> None:
    emp, t = load()
    rows = []
    d50, d75 = pool(emp, t, 0.5), pool(emp, t, 0.75)
    y = emp[TREATED]

    rows.append(evaluate("A. baseline", y, emp[d50], emp, d50, t))
    r50 = pick_ridge(y, emp[d50])
    rows.append(evaluate("B. ridge", y, emp[d50], emp, d50, t, ridge=r50))

    # C. growth: year-over-year change in log employment (4-quarter difference).
    g = emp.diff(4)
    g50 = [d for d in d50 if g[d].loc[2003:].notna().all()]
    rows.append(evaluate("C. growth (yoy)", g[TREATED], g[g50], g, g50, t, growth=True))

    rows.append(evaluate("D. wider pool (<75th pct)", y, emp[d75], emp, d75, t))
    r75 = pick_ridge(y, emp[d75])
    rows.append(evaluate("E. wider pool + ridge", y, emp[d75], emp, d75, t, ridge=r75))

    res = pd.DataFrame(rows).set_index("variant")
    eligible = res[res["effective_donors"] >= MIN_EFFECTIVE_DONORS]
    chosen = eligible["holdout_error_2018_19"].idxmin() if len(eligible) else None
    res.to_csv(RES / "synth_variants.csv")
    (RES / "synth_variants.json").write_text(json.dumps(
        {"selection_rule": f"lowest hold-out error among variants with >= {MIN_EFFECTIVE_DONORS} effective donors",
         "chosen": chosen, "variants": rows}, indent=2))
    pd.set_option("display.width", 250)
    print(res.drop(columns=["ridge"]).T.to_string())
    print(f"\nchosen by the pre-set rule: {chosen}")


if __name__ == "__main__":
    main()
