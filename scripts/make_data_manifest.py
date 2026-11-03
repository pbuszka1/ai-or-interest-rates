"""Record a SHA-256 checksum, size and date for every raw and processed data file.

Anyone rerunning the project can compare their downloads with docs/DATA_MANIFEST.csv: a changed checksum means the
source was revised or the file differs, which explains any change in results. Raw files aren't committed; this records them.
To check your own copy: put your downloads and your IPUMS CPS extract in data/raw (see docs/IPUMS_EXTRACT.md), build the
processed files, then run this script with --check. It lists every file whose checksum differs from the saved manifest.

Usage: python scripts/make_data_manifest.py [--check]   (--check compares current files with the saved manifest)
"""

import csv
import datetime as dt
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "DATA_MANIFEST.csv"


TEXT = {".csv", ".txt", ".json", ".md", ".xml", ".tsv"}


def sha256(p: Path) -> str:
    """SHA-256 of the file. Text files are hashed with Windows line endings (CRLF) turned into LF first, because git
    converts line endings on checkout, so the same file can differ byte for byte between Windows and Linux copies."""
    if p.suffix.lower() in TEXT:
        return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rows() -> list[dict]:
    out = []
    for folder in ["raw", "processed"]:
        for p in sorted((ROOT / "data" / folder).rglob("*")):
            if p.is_file():
                out.append({"file": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p),
                            "modified": dt.datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d")})
    return out


def main() -> None:
    now = rows()
    if "--check" in sys.argv:
        saved = {r["file"]: r["sha256"] for r in csv.DictReader(open(OUT, encoding="utf-8"))}
        changed = [r["file"] for r in now if saved.get(r["file"]) not in (None, r["sha256"])]
        missing = sorted(set(saved) - {r["file"] for r in now})
        print(f"{len(now)} files; changed: {changed or 'none'}; missing: {missing or 'none'}")
        return
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "bytes", "sha256", "modified"])
        w.writeheader()
        w.writerows(now)
    print(f"wrote {OUT.name}: {len(now)} files")


if __name__ == "__main__":
    main()
