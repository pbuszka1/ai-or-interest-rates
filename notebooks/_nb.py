"""Small helpers shared by the analysis notebooks: rerun a script, load its results, show a figure."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from IPython.display import Image, display

ROOT = Path(__file__).resolve().parent.parent


def run(script: str) -> None:
    """Rerun scripts/<script>.py from the processed data and print the end of its output.
    Needs the data/ folder, which isn't in the public repo (see docs/IPUMS_EXTRACT.md)."""
    out = subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "scripts" / f"{script}.py")],
                         cwd=ROOT, capture_output=True, text=True)
    print("\n".join((out.stdout + out.stderr).splitlines()[-15:]))


def results(name: str) -> dict:
    """The saved results/<name>.json."""
    return json.loads((ROOT / "results" / f"{name}.json").read_text())


def figure(name: str) -> None:
    display(Image(filename=str(ROOT / "figures" / f"{name}.png")))


def pct(x, signed: bool = True) -> str:
    """0.0123 -> '+1.2%'."""
    if x is None:
        return "–"
    return f"{100 * x:+.1f}%" if signed else f"{100 * x:.1f}%"


def ci(c) -> str:
    return "–" if not c else f"[{100 * c[0]:+.1f}, {100 * c[1]:+.1f}]"


def show(d: dict) -> pd.DataFrame:
    """A dict of label -> value as a two-column table."""
    return pd.DataFrame({"value": d}).rename_axis("")


def by_year(d: dict) -> pd.DataFrame:
    """{year: effect} -> one-row table of percentages."""
    return pd.DataFrame({str(k): [pct(v)] for k, v in d.items()}, index=["effect"])
