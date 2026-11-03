"""Compare the analysis data with published figures. Needs FRED_API_KEY and data/interim/cps.

CPS by age against BLS, CPS occupations against BLS Table 11, panel coverage, group shares across the
2020 occupation-code change, and CES coverage of total private employment.

Output: results/data_validation.json
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import event_study_industry as ei  # noqa: E402
from build_panel import EMPLOYED, occ2010_to_group  # noqa: E402

# BLS Table 11, annual averages, thousands, all ages 16+. Our CPS extract covers ages 18 to 65, so levels
# are expected to run low; what matters is whether the change from 2019 to 2025 matches.
TABLE_11 = {"Software developers": ([1020], 1815, 2254), "Registered nurses": ([3255], 3242, 3528),
            "Accountants and auditors": ([800], 1964, 1766), "Lawyers": ([2100], 1240, 1146), "Cashiers": ([4720], 3164, 2490),
            "Construction laborers": ([6260], 2051, 2382), "Customer service representatives": ([5240], 2552, 2600),
            "Computer and mathematical (SOC 15)": ("15", 5352, 6711), "Management (SOC 11)": ("11", 18985, 20876)}


def fred(series: str, start: str) -> pd.Series:
    r = requests.get("https://api.stlouisfed.org/fred/series/observations",
                     params={"series_id": series, "api_key": os.environ["FRED_API_KEY"], "file_type": "json", "observation_start": start}, timeout=60)
    r.raise_for_status()
    o = pd.DataFrame(r.json()["observations"])
    return pd.Series(pd.to_numeric(o["value"], errors="coerce").to_numpy(), index=pd.to_datetime(o["date"])).dropna()


def main() -> None:
    load_dotenv(ROOT / ".env")
    d = ds.dataset(ROOT / "data" / "interim" / "cps", format="parquet", partitioning="hive").to_table(
        columns=["YEAR", "MONTH", "AGE", "EMPSTAT", "OCC2010", "WTFINL"], filter=ds.field("YEAR") >= 2010).to_pandas()
    d = d[d["EMPSTAT"].isin(EMPLOYED)]
    d["group"] = d["OCC2010"].map(occ2010_to_group().set_index("OCC2010")["group"])
    out = {"note": "Our CPS extract covers ages 18 to 65."}

    ages = {}
    for label, lo, hi, sid in [("25-54", 25, 54, "LNU02000060"), ("20-24", 20, 24, "LNU02000036")]:
        s = fred(sid, "2010-01-01")
        s.index = pd.MultiIndex.from_arrays([s.index.year, s.index.month], names=["YEAR", "MONTH"])
        ours = d[d["AGE"].between(lo, hi)].groupby(["YEAR", "MONTH"])["WTFINL"].sum() / 1000
        j = pd.concat([ours.rename("o"), s.rename("b")], axis=1).dropna()
        gap = (j["o"] / j["b"] - 1).abs() * 100
        ages[label] = {"series": sid, "months": len(j), "median_abs_gap_pct": float(gap.median()), "max_abs_gap_pct": float(gap.max())}
    out["cps_by_age_vs_bls"] = ages

    occ = {}
    for name, (key, b19, b25) in TABLE_11.items():
        vals = []
        for y in (2019, 2025):
            v = d[d["YEAR"] == y]
            pick = v["group"].fillna("").str.startswith(key) if isinstance(key, str) else v["OCC2010"].isin(key)
            vals.append(v.loc[pick, "WTFINL"].sum() / v["MONTH"].nunique() / 1000)
        occ[name] = {"ours_change_pct": (vals[1] / vals[0] - 1) * 100, "bls_change_pct": (b25 / b19 - 1) * 100}
    out["occupation_change_2019_25_vs_bls_table_11"] = occ

    g = d[d["AGE"].between(18, 65) & d["group"].notna()]
    g = g[~((g["YEAR"] == 2021) & (g["MONTH"] <= 6))]
    e = (g.groupby(["YEAR", "group"])["WTFINL"].sum() / g.groupby("YEAR")["MONTH"].nunique()).unstack("group")
    share = e.div(e.sum(axis=1), axis=0)
    big = share.loc[2019].sort_values(ascending=False).index[:60]
    moves = np.log(share.loc[2021, big] / share.loc[2019, big])
    before = np.log(share.loc[2019, big] / share.loc[2017, big])
    groups = pd.read_csv(ROOT / "data" / "processed" / "occ_groups.csv", dtype={"group": str}).set_index("group")
    x = (groups["dv_rating_beta"] + groups["human_rating_beta"]) / 2
    out["occupation_group_shares"] = {"sd_log_change_2017_19": float(before.std()), "sd_log_change_2019_21": float(moves.std()),
                                      "correlation_with_exposure_2019_21": float(moves.corr(x.reindex(moves.index))),
                                      "largest_moves_2019_21": moves.reindex(moves.abs().sort_values(ascending=False).index).head(6).to_dict()}

    emp, _ = ei.load()
    ids = list(ei.panel(emp, ei.PRIMARY).columns)
    ces = pd.read_parquet(ROOT / "data" / "processed" / "ces_panel.parquet").pivot(index="date", columns="id", values="value")
    ces.index = pd.to_datetime(ces.index)
    share_private = (ces[ids].sum(axis=1) / fred("USPRIV", "2015-01-01")).dropna() * 100
    out["ces_test_a_share_of_total_private_pct"] = {str(y): float(share_private[str(y)].mean()) for y in (2015, 2019, 2022, 2026)}
    path = ROOT / "results" / "data_validation.json"
    path.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
