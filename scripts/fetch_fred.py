"""Download macro and industry series from FRED for the rate and cycle controls.

Saves each series as data/raw/fred/<SERIES_ID>.csv (date, value) plus a manifest with each
series' title, units and last update, so every analysis can say which vintage it used.
Needs FRED_API_KEY in .env.
"""

import datetime as dt
import json
import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "fred"
API = "https://api.stlouisfed.org/fred"

SERIES = {
    # Interest rates
    "FEDFUNDS": "Effective federal funds rate, monthly",
    "GS2": "2-year Treasury yield, monthly",
    "GS10": "10-year Treasury yield, monthly",
    # Prices, for constant-dollar earnings
    "CPIAUCSL": "CPI-U, all items, seasonally adjusted",
    "CPIAUCNS": "CPI-U, all items, not seasonally adjusted",
    # Business cycle
    "UNRATE": "Unemployment rate",
    "USREC": "NBER recession indicator",
    "PAYEMS": "Total nonfarm payroll employment",
    # Tech industry employment (CES)
    "USINFO": "Information sector employment",
    "CES6054150001": "Computer systems design and related services employment",
    "CES6054000001": "Professional, scientific and technical services employment",
    # Job openings (JOLTS, from Dec 2000)
    "JTSJOL": "Job openings, total nonfarm",
    "JTU5100JOL": "Job openings, information (not seasonally adjusted)",
    "JTS540099JOL": "Job openings, professional and business services",
    "JTSHIL": "Hires, total nonfarm",
    "JTU5100HIL": "Hires, information (not seasonally adjusted)",
}


def main() -> None:
    load_dotenv(ROOT / ".env")
    key = os.environ["FRED_API_KEY"]
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"downloaded": dt.date.today().isoformat(), "series": {}}
    for sid, label in SERIES.items():
        params = {"series_id": sid, "api_key": key, "file_type": "json"}
        info = requests.get(f"{API}/series", params=params, timeout=60)
        if info.status_code != 200:
            print(f"{sid:15s} FAILED: {info.json().get('error_message', info.status_code)}")
            continue
        meta = info.json()["seriess"][0]
        obs = requests.get(f"{API}/series/observations", params=params, timeout=60).json()["observations"]
        df = pd.DataFrame(obs)[["date", "value"]]
        df["value"] = pd.to_numeric(df["value"], errors="coerce")  # FRED marks missing as "."
        df.to_csv(OUT / f"{sid}.csv", index=False)
        manifest["series"][sid] = {"label": label, "title": meta["title"], "units": meta["units"],
                                   "frequency": meta["frequency"], "last_updated": meta["last_updated"],
                                   "start": df["date"].iloc[0], "end": df["date"].iloc[-1]}
        print(f"{sid:15s} {df['date'].iloc[0]} to {df['date'].iloc[-1]}  {meta['title'][:60]}")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
