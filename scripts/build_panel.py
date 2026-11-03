"""Build the analysis table: occupation group x age group x quarter, 1990-Q1 to 2026-Q3.

Occupation groups are SOC 2018 *minor groups* (about 100, e.g. 15-12 "computer
occupations"): the standard middle level of the classification, small enough to be distinct
and large enough that each quarter has a usable number of CPS respondents. Each OCC2010 code
joins the minor group holding most of its 2020-2025 employment.

Outputs (data/processed/):
  panel_occ_age_quarter.parquet  one row per group x age group x quarter (the stock outcomes)
  occ_groups.csv                 one row per group: AI exposure, size, 2020-22 boom, labels

Earnings are converted to 2025 dollars with CPI-U (FRED CPIAUCSL), in code rather than by
the IPUMS extract option, so the method is visible.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_exposure_crosswalk import SCORES, census_to_occ2010, census_to_soc  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CPS = ROOT / "data" / "interim" / "cps"
OUT = ROOT / "data" / "processed"

AGE_BINS = [17, 21, 27, 34, 65]
AGE_LABELS = ["18-21", "22-27", "28-34", "35-65"]
EMPLOYED = [10, 12]       # at work; has job, not at work last week
UNEMPLOYED = [20, 21, 22]
COMPUTER_GROUP = "15-12"
COLUMNS = ["YEAR", "MONTH", "AGE", "EMPSTAT", "OCC2010", "WTFINL", "EARNWEEK", "EARNWT"]


def occ2010_to_group() -> pd.DataFrame:
    """Assign each OCC2010 code to the SOC minor group holding most of its employment."""
    census = census_to_soc()[["census2018", "soc"]]
    w = census_to_occ2010().merge(census, on="census2018", how="inner")
    w["group"] = w["soc"].str[:5]
    best = w.groupby(["OCC2010", "group"])["weight"].sum().reset_index() \
            .sort_values("weight").drop_duplicates("OCC2010", keep="last")
    return best[["OCC2010", "group"]]


def occ2010_titles() -> dict[int, str]:
    """OCC2010 code -> title, from the IPUMS codebook (DDI) that came with the extract."""
    import warnings
    from ipumspy import readers
    warnings.filterwarnings("ignore", category=readers.CitationWarning)
    codes = readers.read_ipums_ddi(ROOT / "data" / "raw" / "cps_00001.xml").get_variable_info("OCC2010").codes
    return {code: title for title, code in codes.items()}


def cpi_index() -> pd.Series:
    """CPI-U deflator by (year, month): multiply nominal dollars by this to get 2025 dollars."""
    cpi = pd.read_csv(ROOT / "data" / "raw" / "fred" / "CPIAUCSL.csv", parse_dates=["date"])
    base = cpi.loc[cpi["date"].dt.year == 2025, "value"].mean()
    return pd.Series(base / cpi["value"].values,
                     index=pd.MultiIndex.from_arrays([cpi["date"].dt.year, cpi["date"].dt.month]))


def aggregate_year(df: pd.DataFrame, groups: pd.DataFrame, deflator: pd.Series) -> pd.DataFrame:
    df = df.merge(groups, on="OCC2010", how="left")
    df["group"] = df["group"].fillna("unmapped")
    df["age_group"] = pd.cut(df["AGE"], AGE_BINS, labels=AGE_LABELS)
    df["quarter"] = (df["MONTH"] - 1) // 3 + 1
    emp = df["EMPSTAT"].isin(EMPLOYED)
    df["w_emp"] = np.where(emp, df["WTFINL"], 0.0)
    df["n_emp"] = emp.astype(int)
    df["w_unemp"] = np.where(df["EMPSTAT"].isin(UNEMPLOYED), df["WTFINL"], 0.0)

    # Real weekly earnings: only asked in outgoing rotation months; 9999.99 means not in universe.
    earn = emp & df["EARNWEEK"].between(1, 9999) & (df["EARNWT"] > 0)
    defl = deflator.reindex(pd.MultiIndex.from_arrays([df["YEAR"], df["MONTH"]])).to_numpy()
    df["w_earn"] = np.where(earn, df["EARNWT"], 0.0)
    df["w_logearn"] = np.where(earn, df["EARNWT"] * np.log(df["EARNWEEK"].clip(lower=1) * defl), 0.0)
    df["n_earn"] = earn.astype(int)

    keys = ["YEAR", "quarter", "group", "age_group"]
    g = df.groupby(keys, observed=True)[["w_emp", "n_emp", "w_unemp", "w_earn", "w_logearn", "n_earn"]].sum()
    months = df.groupby(["YEAR", "quarter"])["MONTH"].nunique().rename("months")
    g = g.join(months, on=["YEAR", "quarter"])
    return g.reset_index()


def boom_size(panel: pd.DataFrame) -> pd.Series:
    """Log employment in 2022 above the group's own 2015-2019 linear trend (all ages)."""
    tot = panel.groupby(["group", "t"])["employed"].sum().reset_index()
    tot["log_emp"] = np.log(tot["employed"])
    out = {}
    for grp, d in tot.groupby("group"):
        pre = d[(d["t"] >= 2015) & (d["t"] < 2020)]
        post = d[(d["t"] >= 2022) & (d["t"] < 2023)]
        if len(pre) < 16 or post.empty:
            continue
        slope, intercept = np.polyfit(pre["t"], pre["log_emp"], 1)
        out[grp] = (post["log_emp"] - (intercept + slope * post["t"])).mean()
    return pd.Series(out, name="boom_2022")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    groups, deflator = occ2010_to_group(), cpi_index()
    dataset = ds.dataset(CPS, format="parquet", partitioning="hive")

    parts, pre_weights = [], []
    for year in range(1990, 2027):
        df = dataset.to_table(columns=COLUMNS, filter=ds.field("YEAR") == year).to_pandas()
        parts.append(aggregate_year(df, groups, deflator))
        if 2015 <= year <= 2019:  # pre-period employment by OCC2010, to weight exposure within groups
            e = df[df["EMPSTAT"].isin(EMPLOYED)]
            pre_weights.append(e.groupby("OCC2010")["WTFINL"].sum())
        print(f"{year} done", flush=True)

    p = pd.concat(parts, ignore_index=True)
    p["employed"] = p["w_emp"] / p["months"]          # average monthly employment in the quarter
    p["unemployed"] = p["w_unemp"] / p["months"]
    p["log_real_weekly_earnings"] = p["w_logearn"] / p["w_earn"].replace(0, np.nan)
    p["t"] = p["YEAR"] + (p["quarter"] - 1) / 4
    p = p.rename(columns={"YEAR": "year", "n_emp": "n_respondents_employed", "n_earn": "n_respondents_earnings"})
    p = p[["year", "quarter", "t", "group", "age_group", "employed", "unemployed",
           "log_real_weekly_earnings", "n_respondents_employed", "n_respondents_earnings", "months"]]

    unmapped = p.loc[p["group"] == "unmapped"].groupby("year")["employed"].sum() / p.groupby("year")["employed"].sum()
    print(f"\nemployment in unmapped OCC2010 codes: max {unmapped.max():.2%} (year {unmapped.idxmax()})")
    p = p[p["group"] != "unmapped"]

    # Group traits: exposure weighted by 2015-2019 employment of member occupations.
    exp = pd.read_csv(OUT / "exposure_occ2010.csv").merge(groups, on="OCC2010")
    w = pd.concat(pre_weights).groupby(level=0).sum().rename("pre_weight")
    exp = exp.join(w, on="OCC2010").dropna(subset=["pre_weight"])
    traits = exp.groupby("group").apply(
        lambda d: pd.Series({s: np.average(d[s], weights=d["pre_weight"]) for s in SCORES}), include_groups=False)
    traits["n_occ2010_codes"] = exp.groupby("group").size()
    traits["employed_2015_19"] = p[(p["year"] >= 2015) & (p["year"] <= 2019)].groupby("group")["employed"].sum() / 20
    traits = traits.join(boom_size(p))
    traits["is_computer"] = traits.index == COMPUTER_GROUP
    # Label each group by its largest member occupation.
    biggest = exp.sort_values("pre_weight").drop_duplicates("group", keep="last").set_index("group")["OCC2010"]
    traits["largest_occ2010"] = biggest
    traits["title"] = biggest.map(occ2010_titles())

    p.to_parquet(OUT / "panel_occ_age_quarter.parquet", index=False)
    traits.reset_index().to_csv(OUT / "occ_groups.csv", index=False)

    print(f"panel: {len(p):,} rows, {p['group'].nunique()} groups, {p['t'].nunique()} quarters")
    thin = (p["n_respondents_employed"] < 30).mean()
    print(f"cells with fewer than 30 employed respondents: {thin:.1%}")
    c = p[p["group"] == COMPUTER_GROUP].groupby("year")["employed"].sum() / 4 / 1e3
    print("computer occupations (15-12), avg employment, thousands:")
    print(c.loc[[1990, 1999, 2001, 2002, 2003, 2019, 2022, 2025, 2026]].round(0).to_string())


if __name__ == "__main__":
    main()
