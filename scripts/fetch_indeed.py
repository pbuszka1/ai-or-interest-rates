"""Download Indeed Hiring Lab's US job postings index by occupational category.

Saves the raw CSV to data/raw/ with the download date in the name, so every analysis can
say exactly which snapshot it used. Indeed revises recent values, so never overwrite.
"""

import datetime as dt
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/hiring-lab/job_postings_tracker/master/US/job_postings_by_sector_US.csv"
RAW = Path(__file__).resolve().parent.parent / "data" / "raw"


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    out = RAW / f"indeed_postings_by_sector_US_{dt.date.today():%Y-%m-%d}.csv"
    with urllib.request.urlopen(URL, timeout=60) as resp:
        out.write_bytes(resp.read())
    print(f"saved {out} ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
