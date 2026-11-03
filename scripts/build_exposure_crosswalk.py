"""Attach AI exposure scores (Eloundou et al.) to IPUMS OCC2010 occupation codes.

Three steps, because the scores and the CPS use different occupation systems:
  1. O*NET-SOC detail codes (15-1252.00) -> 6-digit SOC 2018 (15-1252): mean over details.
  2. SOC 2018 -> Census 2018 occupation codes (1021), from the official Census list.
     Some Census codes cover a SOC group (15-12XX, or 41-2010 with trailing zeros); those
     get the mean over the detailed SOC codes in the group.
  3. Census 2018 codes -> OCC2010, built from the CPS itself: since 2020 every worker has
     both codes, so each OCC2010 gets the employment-weighted average of its Census codes.

Output: data/processed/exposure_occ2010.csv, plus a coverage report on how much 2020-2025
employment received a score.
"""

from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
SCORES = ["dv_rating_alpha", "dv_rating_beta", "dv_rating_gamma",
          "human_rating_alpha", "human_rating_beta", "human_rating_gamma"]


def soc_scores() -> pd.DataFrame:
    e = pd.read_csv(RAW / "exposure" / "eloundou_occ_level_2026-10-21.csv")
    e["soc"] = e["O*NET-SOC Code"].str[:7]
    return e.groupby("soc")[SCORES].mean().reset_index()


def census_to_soc() -> pd.DataFrame:
    c = pd.read_excel(RAW / "crosswalks" / "census_2018_occ_code_list_and_crosswalk.xlsx",
                      sheet_name="2018 Census Occ Code List", header=None, skiprows=5)
    c = c.iloc[:, 1:4]
    c.columns = ["title", "census2018", "soc"]
    c = c.dropna(subset=["census2018", "soc"])
    c = c[c["census2018"].astype(str).str.fullmatch(r"\d{4}")]
    c["census2018"] = c["census2018"].astype(int)
    c["soc"] = c["soc"].astype(str).str.strip()
    return c


def census_scores(soc: pd.DataFrame, census: pd.DataFrame) -> pd.DataFrame:
    """Score each Census code: exact SOC match, else the mean over the detailed SOC codes in its group.

    Census codes often point to a SOC group rather than one occupation, written either with
    X (15-12XX) or with trailing zeros (cashiers = 41-2010, covering 41-2011 and 41-2012).
    """
    rows = []
    for _, r in census.iterrows():
        match = soc[soc["soc"] == r["soc"]]
        if match.empty:
            match = soc[soc["soc"].str.startswith(r["soc"].rstrip("X0"))]
        if len(match):
            rows.append({"census2018": r["census2018"], "title": r["title"], "n_soc": len(match),
                         **match[SCORES].mean().to_dict()})
    return pd.DataFrame(rows)


def census_to_occ2010() -> pd.DataFrame:
    """Employment-weighted mapping from Census 2018 codes (CPS OCC, 2020+) to OCC2010."""
    d = ds.dataset(ROOT / "data" / "interim" / "cps", format="parquet", partitioning="hive")
    t = d.to_table(columns=["OCC", "OCC2010", "WTFINL", "EMPSTAT"],
                   filter=(ds.field("YEAR") >= 2020) & (ds.field("YEAR") <= 2025)).to_pandas()
    t = t[t["EMPSTAT"].isin([10, 12])]
    return t.groupby(["OCC", "OCC2010"])["WTFINL"].sum().rename("weight").reset_index() \
            .rename(columns={"OCC": "census2018"})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cs = census_scores(soc_scores(), census_to_soc())
    m = census_to_occ2010().merge(cs, on="census2018", how="left")

    covered = m["dv_rating_beta"].notna()
    print(f"2020-2025 employment with a score: {m.loc[covered, 'weight'].sum() / m['weight'].sum():.1%}")
    unmatched = m[~covered].groupby("census2018")["weight"].sum().sort_values(ascending=False)
    print("largest unmatched Census codes (share of employment):")
    print((unmatched.head(8) / m["weight"].sum()).round(4).to_string())

    m = m[covered]
    w = m.groupby("OCC2010")["weight"].transform("sum")
    for s in SCORES:
        m[s] = m[s] * m["weight"] / w
    occ = m.groupby("OCC2010")[SCORES].sum()
    occ["employment_weight"] = m.groupby("OCC2010")["weight"].sum()
    occ.reset_index().to_csv(OUT / "exposure_occ2010.csv", index=False)
    print(f"wrote {len(occ)} OCC2010 occupations to {OUT / 'exposure_occ2010.csv'}")

    print("\nspot check (dv = GPT-4 rating, human = annotator rating):")
    check = {1020: "Software developers", 1006: "Computer scientists/systems analysts",
             1010: "Programmers", 800: "Accountants", 3255: "Registered nurses",
             6260: "Construction laborers", 4020: "Cooks", 5240: "Customer service reps"}
    print(occ.loc[occ.index.intersection(list(check)), ["dv_rating_beta", "human_rating_beta"]]
             .rename(index=check).round(2).to_string())


if __name__ == "__main__":
    main()
