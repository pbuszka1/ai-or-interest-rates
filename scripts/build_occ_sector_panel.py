"""Build the Test B table: occupation group x industry sector x year, CPS 2011-2026.

One row per cell and age band ("18-65", "22-27", "35-65"), with average monthly weighted
employment and the number of respondent-months behind it. Months from 2020-01 to 2021-06 (the
pandemic) are dropped, so 2021 is July-December only and 2020 is absent.

Sectors come from IND1990, IPUMS's harmonized industry code, which stays consistent across the
Census industry recodes since 2003. The information and professional-services industries (where
AI-exposed occupations cluster: telecom, publishing, computer services, legal, accounting,
engineering, consulting, research, advertising) are carved out as their own sector, so Test B's
robustness check can drop them.

Writes data/processed/panel_occ_sector_year.parquet.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_panel import EMPLOYED, occ2010_to_group  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CPS, OUT = ROOT / "data" / "interim" / "cps", ROOT / "data" / "processed"
COLUMNS = ["YEAR", "MONTH", "AGE", "EMPSTAT", "OCC2010", "IND1990", "WTFINL"]
AGE_BANDS = {"18-65": (18, 65), "22-27": (22, 27), "35-65": (35, 65)}

# IND1990 code ranges -> sector. Checked in order, so the carve-out comes first.
INFO_PROF = {171, 172, 440, 441, 442, 721, 732, 841, 882, 890, 891, 892, 893}
SECTORS = [("agriculture", 10, 32), ("mining", 40, 50), ("construction", 60, 60), ("manufacturing", 100, 392),
           ("transport_utilities", 400, 472), ("wholesale", 500, 571), ("retail", 580, 691), ("finance", 700, 712),
           ("business_repair", 721, 760), ("personal", 761, 791), ("entertainment", 800, 810),
           ("professional_related", 812, 893), ("public_admin", 900, 932)]


def sector(ind: pd.Series) -> pd.Series:
    out = pd.Series(pd.NA, index=ind.index, dtype="object")
    for name, lo, hi in SECTORS:
        out[ind.between(lo, hi)] = name
    out[ind.isin(INFO_PROF)] = "info_professional"
    return out                                            # military and not-in-universe stay NA


def main() -> None:
    groups = occ2010_to_group()
    data = ds.dataset(CPS, format="parquet", partitioning="hive")
    parts = []
    for year in range(2011, 2027):
        df = data.to_table(columns=COLUMNS, filter=ds.field("YEAR") == year).to_pandas()
        df = df[~((df["YEAR"] == 2020) | ((df["YEAR"] == 2021) & (df["MONTH"] <= 6)))]
        if df.empty:
            continue
        months = df["MONTH"].nunique()                    # e.g. 2025 has 11 (no October survey)
        df = df[df["EMPSTAT"].isin(EMPLOYED)].merge(groups, on="OCC2010", how="inner")
        df["sector"] = sector(df["IND1990"])
        df = df.dropna(subset=["sector"])
        for band, (lo, hi) in AGE_BANDS.items():
            d = df[df["AGE"].between(lo, hi)]
            g = d.groupby(["group", "sector"]).agg(employed=("WTFINL", "sum"), n_resp_months=("WTFINL", "size"))
            g["employed"] /= months                       # average monthly employment
            parts.append(g.reset_index().assign(year=year, age_band=band, months=months))
        print(f"{year}: {months} months", flush=True)
    p = pd.concat(parts, ignore_index=True)
    p.to_parquet(OUT / "panel_occ_sector_year.parquet", index=False)
    a = p[p["age_band"] == "18-65"]
    print(f"{len(p):,} rows; all ages: {a.groupby(['group', 'sector']).ngroups} cells, "
          f"{a['group'].nunique()} groups, {a['sector'].nunique()} sectors")
    print("median respondent-months per cell-year (all ages):", int(np.median(a["n_resp_months"])))


if __name__ == "__main__":
    main()
