"""Sanity checks on the Parquet CPS panel before any analysis touches it.

1. Every expected month is present (Jan 1990 to Sep 2026, except Oct 2025) and none twice.
2. No ASEC supplement records survived.
3. Ages are within the extract's 18-65 selection. People outside the labor force are kept on
   purpose (see the comment in main), so this only reports their counts.
4. Weighted employment per month is in a plausible range for US workers aged 18-65
   (BLS reports about 120 to 155 million employed of all ages over this period).
5. Software developers (OCC2010 1020) have a sample count large enough to analyse. Note that
   this code only exists from 2003: before then software developers were coded as computer
   scientists and systems analysts (1006) or programmers (1010). See the project log.

Prints a report and exits non-zero if a hard check fails.
"""

import sys
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parent.parent
CPS = ROOT / "data" / "interim" / "cps"
SOFTWARE_DEVELOPERS = 1020  # OCC2010: "Software developers, applications and systems software"


def main() -> int:
    cols = ["YEAR", "MONTH", "AGE", "LABFORCE", "EMPSTAT", "ASECFLAG", "WTFINL", "OCC2010"]
    df = ds.dataset(CPS, format="parquet", partitioning="hive").to_table(columns=cols).to_pandas()
    failures = []

    months = df.groupby(["YEAR", "MONTH"]).size()
    expected = pd.period_range("1990-01", "2026-09", freq="M").difference(pd.PeriodIndex(["2025-10"], freq="M"))
    found = pd.PeriodIndex([pd.Period(year=int(y), month=int(m), freq="M") for y, m in months.index])
    missing, extra = expected.difference(found), found.difference(expected)
    print(f"rows: {len(df):,} | months: {len(found)} (expected {len(expected)})")
    if len(missing) or len(extra):
        failures.append(f"missing months {list(missing)[:5]}, unexpected {list(extra)[:5]}")
    print(f"rows per month: min {months.min():,}, median {int(months.median()):,}, max {months.max():,}")

    if (df["ASECFLAG"].fillna(2) == 1).any():
        failures.append("ASEC records present")
    if df["AGE"].min() < 18 or df["AGE"].max() > 65:
        failures.append(f"ages outside 18-65: {df['AGE'].min()}-{df['AGE'].max()}")
    # People outside the labor force are deliberately kept (LABFORCE 1 = not in labor force,
    # 0 = not in universe): filtering on labor-force status would condition on an outcome the
    # shocks themselves can change. Analyses filter to LABFORCE == 2 only where they need to.
    print("LABFORCE counts (0 NIU, 1 not in LF, 2 in LF):", df["LABFORCE"].value_counts().sort_index().to_dict())

    # EMPSTAT 10 = at work, 12 = has job, not at work last week.
    employed = df[df["EMPSTAT"].isin([10, 12])]
    emp = employed.groupby(["YEAR", "MONTH"])["WTFINL"].sum() / 1e6
    print(f"weighted employed 18-65 (millions): min {emp.min():.1f}, max {emp.max():.1f}")
    print(emp.groupby(level=0).mean().round(1).loc[[1990, 2000, 2001, 2003, 2008, 2010, 2019, 2020, 2022, 2024, 2026]].to_string())
    if emp.min() < 90 or emp.max() > 175:
        failures.append("weighted employment outside plausible range")

    # Average over the months actually present: 2025 has 11 (no October) and 2026 has 9.
    # Dividing a year's total by 12 understates those years; that bug was caught on 2026-10-22.
    sw = employed[employed["OCC2010"] == SOFTWARE_DEVELOPERS]
    sd = pd.DataFrame({
        "sample": sw.groupby("YEAR").size(),
        "weighted_thousands": sw.groupby(["YEAR", "MONTH"])["WTFINL"].sum().groupby(level=0).mean() / 1e3,
    })
    print("\nsoftware developers (OCC2010 1020), per year: sample size and avg weighted employment (thousands)")
    print(sd.round(0).to_string())

    print("\nFAILED: " + "; ".join(failures) if failures else "\nall hard checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
