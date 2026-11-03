"""Industry-level panel from CES payroll employment, with AI exposure and rate sensitivity.

Why industries: the CES surveys about 120,000 businesses a month, so industry employment is
far less noisy than CPS occupation groups (where rate sensitivity was undetectable and the
first synthetic control failed its hold-out check; see the project log, 2026-10-22).

Steps:
  1. Take FRED's seasonally adjusted "all employees" CES series (BLS blocks scripted downloads
     without a contact email, so FRED is the route) and keep the *leaves*: the most detailed
     industries, so no donor contains another.
  2. Download each leaf's monthly employment from FRED.
  3. AI exposure of an industry = employment-weighted exposure of the occupations it employs
     (CPS 2015-2019, as pre-registered; Census 2012 industry codes are updated to 2017 codes
     using the Census list of changes, then mapped to NAICS with the Census 2017 crosswalk).
  4. Rate sensitivity and cyclicality per industry, 1990-2019, same method as
     scripts/estimate_rate_sensitivity.py, to test whether larger samples make it measurable.

Outputs: data/raw/ces/series/<id>.csv, data/processed/ces_panel.parquet,
         data/processed/ces_industries.csv
"""

import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import requests
import statsmodels.api as sm
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from estimate_rate_sensitivity import LAGS, quarterly_shocks, shrink  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
AGGREGATE_SUPERSECTORS = {"00", "05", "06", "07", "08"}  # total, private, goods, services groupings
# Government (supersector 90) doesn't hire like a market employer and has no CPS exposure
# match; 65620001 "Health Care" is a special BLS aggregate whose code breaks the NAICS
# pattern, so the leaf test mistakes it for a detailed industry though it contains hospitals etc.
EXCLUDE_SUPERSECTORS, EXCLUDE_CODES = {"90"}, {"65620001"}


def leaves() -> pd.DataFrame:
    s = pd.read_csv(RAW / "ces" / "fred_ces_all_employees_series.csv", dtype={"code": str})
    s["ss"], s["prefix"] = s["code"].str[:2], s["code"].str[2:].str.rstrip("0")
    s = s[~s["ss"].isin(AGGREGATE_SUPERSECTORS | EXCLUDE_SUPERSECTORS) & ~s["code"].isin(EXCLUDE_CODES)
          & (s["prefix"] != "") & (s["observation_start"] <= "1990-01-01")]
    has_child = s.apply(lambda r: ((s["ss"] == r["ss"]) & (s["prefix"] != r["prefix"])
                                   & s["prefix"].str.startswith(r["prefix"])).any(), axis=1)
    return s[~has_child].reset_index(drop=True)


def download(ids: list[str]) -> pd.DataFrame:
    load_dotenv(ROOT / ".env")
    key, out = os.environ["FRED_API_KEY"], RAW / "ces" / "series"
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    for sid in ids:
        path = out / f"{sid}.csv"
        if not path.exists():
            obs = requests.get("https://api.stlouisfed.org/fred/series/observations",
                               params={"series_id": sid, "api_key": key, "file_type": "json"}, timeout=60).json()
            pd.DataFrame(obs["observations"])[["date", "value"]].to_csv(path, index=False)
            time.sleep(0.6)  # FRED allows 120 requests a minute
        d = pd.read_csv(path, parse_dates=["date"])
        d["value"], d["id"] = pd.to_numeric(d["value"], errors="coerce"), sid
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


# Census industry codes that changed between the 2012 and 2017 lists, from the "Summary of Changes"
# sheet of the crosswalk workbook. CPS uses 2012 codes through 2019 and 2017 codes from 2020.
CODE_CHANGES_2012_TO_2017 = {
    1680: [1691], 1690: [1691], 3190: [3291], 3290: [3291], 4970: [4971, 4972], 5380: [5381, 5391], 5390: [5391],
    5590: [5593], 5591: [5593], 5592: [5593], 6990: [6991, 6992], 7070: [7071, 7072], 7170: [7181], 7180: [7181],
    8190: [8191, 8192], 8560: [8561, 8562, 8563, 8564], 8880: [8891], 8890: [8891]}


def recode_2012_industries(cps: pd.DataFrame) -> pd.DataFrame:
    """Move 2012 industry codes to 2017 codes. A split code shares its weight evenly among its children,
    which each take the parent's occupation mix, so total weight is unchanged."""
    out = cps.copy()
    out["IND"] = out["IND"].map(lambda i: CODE_CHANGES_2012_TO_2017.get(int(i), [int(i)]))
    out["WTFINL"] = out["WTFINL"] / out["IND"].map(len)
    out = out.explode("IND")
    out["IND"] = out["IND"].astype(int)
    return out


# CES aggregate codes that are not NAICS codes, mapped to the NAICS industries they combine.
CES_AGGREGATES = {"336001": ["3361", "3362", "3363"], "329": ["312", "316"]}


def industry_exposure(leaf: pd.DataFrame, first: int = 2015, last: int = 2019) -> pd.DataFrame:
    """Employment-weighted occupational exposure for each CES leaf, via Census industry codes.

    Defaults to the pre-registered 2015-19 occupation mix. Other windows are sensitivity checks.
    "Not specified" catch-all Census rows (NAICS listed as "Part of ...") are used only when an industry has no
    specific match, so they don't blend a sector average into every industry in that sector.
    """
    xw = pd.read_excel(RAW / "crosswalks" / "census_2017_industry_code_list_with_crosswalk.xlsx",
                       sheet_name="2017 Census Industry Code List", header=None, skiprows=4).iloc[:, [3, 4]]
    xw.columns = ["IND", "naics"]
    xw = xw[pd.to_numeric(xw["IND"], errors="coerce").notna()].astype({"IND": int}).drop_duplicates("IND")
    xw["catch_all"] = xw["naics"].astype(str).str.strip().str.startswith("Part of")
    xw["naics"] = xw["naics"].astype(str).apply(lambda v: [re.sub(r"\D", "", n) for n in v.split(",") if re.sub(r"\D", "", n)])

    occ = pd.read_csv(PROC / "exposure_occ2010.csv")[["OCC2010", "dv_rating_beta", "human_rating_beta"]]
    cps = ds.dataset(ROOT / "data" / "interim" / "cps", format="parquet", partitioning="hive").to_table(
        columns=["IND", "OCC2010", "EMPSTAT", "WTFINL"],
        filter=(ds.field("YEAR") >= first) & (ds.field("YEAR") <= last)).to_pandas()
    cps = cps[cps["EMPSTAT"].isin([10, 12])].merge(occ, on="OCC2010")
    if last <= 2019:
        cps = recode_2012_industries(cps)
    by_ind = cps.groupby("IND").apply(lambda d: pd.Series({
        "w": d["WTFINL"].sum(),
        "dv_rating_beta": np.average(d["dv_rating_beta"], weights=d["WTFINL"]),
        "human_rating_beta": np.average(d["human_rating_beta"], weights=d["WTFINL"])}), include_groups=False)
    ind = xw.join(by_ind, on="IND").dropna(subset=["w"])

    rows = []
    for _, lf in leaf.iterrows():
        ps = CES_AGGREGATES.get(lf["prefix"], [lf["prefix"]])
        match = ind[ind["naics"].apply(lambda ns: any(n.startswith(p) or p.startswith(n) for n in ns for p in ps))]
        if match.empty:
            continue
        if (~match["catch_all"]).any():
            match = match[~match["catch_all"]]
        rows.append({"id": lf["id"], "n_census_industries": len(match),
                     "dv_rating_beta": np.average(match["dv_rating_beta"], weights=match["w"]),
                     "human_rating_beta": np.average(match["human_rating_beta"], weights=match["w"])})
    return pd.DataFrame(rows)


def rate_traits(q: pd.DataFrame) -> pd.DataFrame:
    """Rate sensitivity and cyclicality per industry, 1990-2019 (no CES coding breaks to drop)."""
    q = q[(q.index >= 1990) & (q.index < 2020)]
    dlog, dall = np.log(q).diff(), np.log(q.sum(axis=1)).diff()
    X = pd.concat({f"s{k}": quarterly_shocks("mps").reindex(q.index).shift(k) for k in range(LAGS)}, axis=1)
    ok = X.notna().all(axis=1) & dall.notna()
    cols, rows = [f"s{k}" for k in range(LAGS)], []
    for c in q.columns:
        use = ok & dlog[c].notna()  # a few series have gaps; drop those quarters for that industry
        if use.sum() < 60:
            continue
        y = (dlog[c] - dall)[use]
        f = sm.OLS(y, sm.add_constant(X[use])).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
        cy = sm.OLS(dlog[c][use], sm.add_constant(dall[use])).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
        V = f.cov_params().loc[cols, cols].to_numpy()
        rows.append({"id": c, "rate_sensitivity_raw": f.params[cols].sum(),
                     "rate_sensitivity_se": float(np.sqrt(np.ones(LAGS) @ V @ np.ones(LAGS))),
                     "cyclicality_raw": cy.params.iloc[1], "cyclicality_se": cy.bse.iloc[1]})
    r = pd.DataFrame(rows).set_index("id")
    r["rate_sensitivity"] = shrink(r["rate_sensitivity_raw"], r["rate_sensitivity_se"])
    r["cyclicality"] = shrink(r["cyclicality_raw"], r["cyclicality_se"])
    return r


def main() -> None:
    leaf = leaves()
    print(f"CES leaf industries starting by 1990: {len(leaf)}")
    panel = download(leaf["id"].tolist())
    panel.to_parquet(PROC / "ces_panel.parquet", index=False)

    m = panel.pivot(index="date", columns="id", values="value")
    q = m.groupby(m.index.year + (m.index.month - 1) // 3 / 4).mean()  # quarterly averages
    q = q[q.index < 2026.75]

    traits = leaf.set_index("id")[["code", "title"]]
    traits["title"] = traits["title"].str.replace("All Employees, ", "", regex=False)
    traits["employed_2015_19"] = m.loc["2015":"2019"].mean() * 1000  # CES is in thousands
    traits = traits.join(industry_exposure(leaf).set_index("id")).join(rate_traits(q))
    traits.reset_index().to_csv(PROC / "ces_industries.csv", index=False)

    r = traits.dropna(subset=["rate_sensitivity_raw"])
    sig = (r["rate_sensitivity_raw"].abs() > 1.96 * r["rate_sensitivity_se"]).mean()
    print(f"exposure found for {traits['dv_rating_beta'].notna().sum()} of {len(traits)} industries")
    print(f"rate sensitivity significant at 5%: {sig:.0%} of {len(r)} industries "
          f"(sd of estimates {r['rate_sensitivity_raw'].std():.3f} vs median s.e. {r['rate_sensitivity_se'].median():.3f})")
    print(f"correlation of exposure (GPT-4) with rate sensitivity {traits['dv_rating_beta'].corr(traits['rate_sensitivity']):.2f}, "
          f"with cyclicality {traits['dv_rating_beta'].corr(traits['cyclicality']):.2f}")
    show = traits.sort_values("rate_sensitivity_raw")[["title", "dv_rating_beta", "rate_sensitivity_raw",
                                                        "rate_sensitivity_se", "cyclicality"]]
    print("\nmost rate-sensitive (negative = shrinks after a tightening surprise):")
    print(show.head(6).round(3).to_string())
    print("\ncomputer systems design:")
    print(traits.loc[["CES6054150001"], ["title", "dv_rating_beta", "human_rating_beta", "rate_sensitivity_raw",
                                         "rate_sensitivity_se", "cyclicality"]].round(3).to_string())


if __name__ == "__main__":
    main()
