"""Exploratory, not pre-registered: does Test B's cell filter affect the answer?

Test B keeps cells with >= 30 respondent-months in every year, including years after ChatGPT.
That filter can drop cells that shrank, so it selects partly on the outcome. This check applies
the same threshold to years through 2019 only, then reruns Test B and its age version.

Output: results/test_b_filter_check.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import event_study as es  # noqa: E402
import event_study_occupation as eo  # noqa: E402

PROC = ROOT / "data" / "processed"
LAST_PRE_YEAR = 2019


def load_band(raw: pd.DataFrame, band: str, pre_only: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Log employment (years x cells) and each cell's group, sector and weight."""
    p = raw[raw["age_band"] == band]
    years = sorted(p["year"].unique())
    use = [y for y in years if y <= LAST_PRE_YEAR] if pre_only else years
    ok = p[(p["n_resp_months"] >= eo.MIN_RESP) & p["year"].isin(use)].groupby(["group", "sector"])["year"].nunique() == len(use)
    q = p.set_index(["group", "sector"]).loc[ok[ok].index].reset_index()
    q = q[q["employed"] > 0]
    full = q.groupby(["group", "sector"])["year"].nunique() == len(years)  # the log needs a value every year
    q = q.set_index(["group", "sector"]).loc[full[full].index].reset_index()
    q["cell"] = q["group"] + "|" + q["sector"]
    y = np.log(q.pivot(index="year", columns="cell", values="employed"))
    y.index = y.index.astype(float)
    cells = q.drop_duplicates("cell").set_index("cell")[["group", "sector"]]
    cells["weight"] = q[q["year"].between(2015, 2019)].groupby("cell")["employed"].mean()
    return y, cells


def all_ages(raw: pd.DataFrame, traits: pd.DataFrame, pre_only: bool) -> dict:
    y, cells = load_band(raw, "18-65", pre_only)
    _, r = eo.estimate(y, cells, eo.group_exposure(traits, cells["group"]), traits)
    return {"n_cells": r["n_cells"], "delta_top_vs_bottom_quartile": r["delta_top_vs_bottom_quartile"], "pretrend_p": r["pretrend"]["p"]}


def age_version(raw: pd.DataFrame, traits: pd.DataFrame, pre_only: bool) -> dict:
    ys, cs = [], []
    for band, young in [("22-27", 1.0), ("35-65", 0.0)]:
        y, cells = load_band(raw, band, pre_only)
        ys.append(y.add_suffix(f"|{band}"))
        cs.append(cells.set_axis(cells.index + f"|{band}").assign(young=young, block=(cells["sector"] + f"|{band}").to_numpy()))
    y, cells = pd.concat(ys, axis=1).dropna(axis=1), pd.concat(cs)
    cells = cells.loc[y.columns]
    xg = eo.group_exposure(traits, cells["group"])
    x = cells["group"].map(xg)
    f = es.fit(y, {"X": x, "XY": x * cells["young"]}, eo.REF, block=cells["block"], cluster=cells["group"], weight=cells["weight"])
    q = es.quartile_gap(xg)
    return {
        "n_cells": y.shape[1],
        "n_young_cells": int((cells["young"] == 1).sum()),
        "delta_2024_2026": es.test(f, es.contrast(f, eo.POST, eo.BASE, "XY"), q),
        "delta_2024_2025": es.test(f, es.contrast(f, [2024, 2025], eo.BASE, "XY"), q),
    }


def main() -> None:
    raw = pd.read_parquet(PROC / "panel_occ_sector_year.parquet")
    traits = pd.read_csv(PROC / "occ_groups.csv", dtype={"group": str}).set_index("group") \
        .join(pd.read_csv(PROC / "occ_rate_sensitivity.csv", dtype={"group": str}).set_index("group")["cyclicality"])
    out = {"note": "Exploratory, not pre-registered, not corrected for multiple testing."}
    for name, pre in [("every_year_filter", False), ("filter_through_2019", True)]:
        out[name] = {"all_ages": all_ages(raw, traits, pre), "age_version": age_version(raw, traits, pre)}
    path = ROOT / "results" / "test_b_filter_check.json"
    path.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
