"""Convert the IPUMS CPS fixed-width extract into Parquet files, one per year.

Reads the gzipped .dat file in chunks using the DDI codebook (so it never has to be unzipped
or held in memory at once), drops ASEC supplement records so March isn't double-counted, and
stores compact integer types. Output: data/interim/cps/year=YYYY/part-NNN.parquet.

Usage: .venv/Scripts/python.exe scripts/load_cps.py [extract_number]
"""

import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from ipumspy import readers

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "interim" / "cps"
CHUNK_ROWS = 1_000_000

# Weights and money stay as floats; everything else is a small integer code.
FLOAT_COLS = {"HWTFINL", "WTFINL", "EARNWT", "LNKFW1MWT", "HOURWAGE2", "EARNWEEK"}
ID_COLS = {"SERIAL", "CPSID", "CPSIDP", "CPSIDV"}


def compact(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast integer codes to the smallest type that holds them."""
    for col in df.columns:
        if col in FLOAT_COLS:
            df[col] = df[col].astype("float32" if col != "EARNWEEK" else "float64")
        elif col in ID_COLS:
            df[col] = df[col].astype("int64")
        else:
            df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def main(extract: str = "00001") -> None:
    warnings.filterwarnings("ignore", category=readers.CitationWarning)
    ddi = readers.read_ipums_ddi(RAW / f"cps_{extract}.xml")
    OUT.mkdir(parents=True, exist_ok=True)

    start, total, kept, part = time.time(), 0, 0, 0
    for chunk in readers.read_microdata_chunked(ddi, RAW / f"cps_{extract}.dat.gz", chunksize=CHUNK_ROWS):
        total += len(chunk)
        # ASECFLAG == 1 marks March supplement records; Basic Monthly records are 2 or missing.
        chunk = chunk[chunk["ASECFLAG"].fillna(2) != 1]
        chunk = compact(chunk)
        kept += len(chunk)
        for year, rows in chunk.groupby("YEAR"):
            target = OUT / f"year={year}"
            target.mkdir(exist_ok=True)
            pq.write_table(pa.Table.from_pandas(rows, preserve_index=False), target / f"part-{part:04d}.parquet")
        part += 1
        print(f"{total:>12,} rows read | {kept:>12,} kept | {time.time() - start:7.0f}s", flush=True)

    print(f"done: {kept:,} of {total:,} rows kept in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main(*sys.argv[1:])
