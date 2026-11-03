"""Download Census QWI (Quarterly Workforce Indicators) by state, industry and age group.

QWI comes from state unemployment-insurance records, so it counts actual hires, not survey
answers. It's reported by industry (NAICS 2022), not occupation. Each state joined the
program in a different year (Washington from 1990, New York from 2000), so coverage varies.

Industries are chosen in three groups to match docs/METHODOLOGY.md:
  - tech and other high-AI-exposure services,
  - interest-rate-sensitive industries with low AI exposure,
  - industries sensitive to neither (low-exposure baseline).

Output: data/raw/qwi/qwi_<state>.csv, one file per state, plus a manifest. Re-running
skips states already downloaded. Needs CENSUS_API_KEY in .env.
"""

import csv
import datetime as dt
import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "qwi"
BASE = "https://api.census.gov/data/timeseries/qwi/sa"

# Emp: employment at start of quarter; EmpS: stable (full-quarter) employment;
# HirA: all hires; HirN: new hires (no recent tie to the employer); Sep: separations;
# EarnS: avg monthly earnings of stable workers; EarnHirNS: avg monthly earnings of new hires.
VARIABLES = ["Emp", "EmpS", "HirA", "HirN", "Sep", "EarnS", "EarnHirNS"]

INDUSTRIES = {
    # High AI exposure, tech and professional services
    "5415": "Computer systems design and related services",
    "5132": "Software publishers",
    "5182": "Computing infrastructure, data processing and web hosting",
    "5192": "Web search portals, libraries, archives and other information services",
    "5417": "Scientific research and development services",
    "5416": "Management, scientific and technical consulting",
    "5412": "Accounting, tax preparation, bookkeeping and payroll",
    "5411": "Legal services",
    "5418": "Advertising, public relations and related services",
    "5221": "Depository credit intermediation",
    "5239": "Other financial investment activities",
    "5241": "Insurance carriers",
    "5613": "Employment services",
    # Interest-rate sensitive, low AI exposure
    "2361": "Residential building construction",
    "2382": "Building equipment contractors",
    "5311": "Lessors of real estate",
    "5312": "Offices of real estate agents and brokers",
    "4411": "Automobile dealers",
    "4441": "Building material and supplies dealers",
    "3361": "Motor vehicle manufacturing",
    "3363": "Motor vehicle parts manufacturing",
    # Low exposure, low rate sensitivity
    "6211": "Offices of physicians",
    "6221": "General medical and surgical hospitals",
    "6231": "Nursing care facilities",
    "7225": "Restaurants and other eating places",
    "5617": "Services to buildings and dwellings",
    "8111": "Automotive repair and maintenance",
}

STATES = [f"{s:02d}" for s in [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                               24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
                               42, 44, 45, 46, 47, 48, 49, 50, 51, 53, 54, 55, 56]]


def get(params: dict, key: str, tries: int = 4) -> list[list[str]]:
    """One API call with retries; returns rows including the header, or [] if no data."""
    for attempt in range(tries):
        r = requests.get(BASE, params={**params, "key": key}, timeout=180)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 204:  # no data for this combination
            return []
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")


def main() -> None:
    load_dotenv(ROOT / ".env")
    key = os.environ["CENSUS_API_KEY"]
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    for state in STATES:
        path = OUT / f"qwi_{state}.csv"
        if path.exists():
            continue
        rows, header = [], None
        for industry in INDUSTRIES:
            data = get({"get": ",".join(VARIABLES + ["agegrp"]), "for": f"state:{state}",
                        "time": "from 1990-Q1 to 2026-Q4", "industry": industry}, key)
            if not data:
                continue
            header = header or data[0]
            rows.extend(data[1:])
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            if header:
                w.writerow(header)
                w.writerows(rows)
        print(f"state {state}: {len(rows):>7,} rows | {time.time() - start:6.0f}s", flush=True)

    manifest = {"downloaded": dt.date.today().isoformat(), "endpoint": BASE, "variables": VARIABLES,
                "industries": INDUSTRIES, "states": STATES, "time": "1990-Q1 to 2026-Q4",
                "note": "seasonally adjusted (sa) series; all age groups; ownership: API default (all owners)"}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"done in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
