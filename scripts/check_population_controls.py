"""Do the CPS's January population-control updates move the young-worker comparison?

Each January, Census re-benchmarks CPS weights to new population estimates, by age, sex and race. That shifts
employment levels by age between December and January. The triple difference compares the exposure gradient for
young workers with the gradient for older workers, so a shift that hits all occupations alike within an age band
cancels. This checks whether it does: the month-to-month change in the young-minus-older exposure gap should be no
larger in January than in other months.

Output: results/population_controls_check.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import event_study_occupation as eo  # noqa: E402
from build_panel import EMPLOYED, occ2010_to_group  # noqa: E402


def main() -> None:
    traits = pd.read_csv(ROOT / "data" / "processed" / "occ_groups.csv", dtype={"group": str}).set_index("group")
    x = eo.group_exposure(traits, traits.index)
    top, bottom = set(x.index[x >= x.quantile(0.75)]), set(x.index[x <= x.quantile(0.25)])
    gm = occ2010_to_group().set_index("OCC2010")["group"]
    d = ds.dataset(ROOT / "data" / "interim" / "cps", format="parquet", partitioning="hive").to_table(
        columns=["YEAR", "MONTH", "AGE", "EMPSTAT", "OCC2010", "WTFINL"], filter=ds.field("YEAR") >= 2011).to_pandas()
    d = d[d["EMPSTAT"].isin(EMPLOYED)].copy()
    d["group"] = d["OCC2010"].map(gm)
    d["q"] = np.where(d["group"].isin(top), "top", np.where(d["group"].isin(bottom), "bottom", None))
    d["band"] = np.where(d["AGE"].between(22, 27), "young", np.where(d["AGE"].between(35, 65), "older", None))
    d = d.dropna(subset=["q", "band"])
    m = d.groupby(["YEAR", "MONTH", "band", "q"])["WTFINL"].sum().unstack(["band", "q"])
    lm = np.log(m)
    gap = (lm[("young", "top")] - lm[("young", "bottom")]) - (lm[("older", "top")] - lm[("older", "bottom")])
    level = np.log(m.T.groupby(level="band").sum().T)
    ch, lch = gap.diff().dropna(), level.diff().dropna()
    months = ch.index.get_level_values("MONTH")
    jan, other = ch[months == 1], ch[months != 1]
    out = {"young_minus_older_exposure_gap_monthly_change": {"january_mean": float(jan.mean()), "january_sd": float(jan.std()), "other_mean": float(other.mean()),
                                                             "other_sd": float(other.std()), "january_by_year": {int(k[0]): float(v) for k, v in jan.items()}},
           "young_employment_monthly_log_change": {"january_by_year": {int(k[0]): float(v) for k, v in lch["young"][lch.index.get_level_values("MONTH") == 1].items()},
                                                   "other_sd": float(lch["young"][lch.index.get_level_values("MONTH") != 1].std())}}
    (ROOT / "results" / "population_controls_check.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
