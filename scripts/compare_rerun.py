"""Compare a rerun of the project with the saved results, number by number.

Run the whole project in a fresh copy (clone the repository, make a new environment from requirements.txt, add the raw
data, then scripts/load_cps.py and scripts/run_all.py), and point this script at that copy. It flattens every
results/*.json file the rerun regenerated into its numbers and compares each with the saved value here.

Usage: python scripts/compare_rerun.py <path to the rerun copy> [--save]
       --save writes the summary to results/rerun_comparison.json
"""

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOL = 1e-9


def numbers(o, prefix: str = "") -> dict[str, float]:
    """Every number in a JSON object, keyed by its path."""
    out: dict[str, float] = {}
    if isinstance(o, dict):
        for k, v in o.items():
            out |= numbers(v, f"{prefix}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            out |= numbers(v, f"{prefix}[{i}]")
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        out[prefix] = float(o)
    return out


def main() -> None:
    rerun = Path(sys.argv[1]).resolve()
    commit = subprocess.run(["git", "log", "--format=%h", "-1"], cwd=rerun, capture_output=True, text=True).stdout.strip()
    since = (rerun / "requirements.txt").stat().st_mtime + 60   # checkout time: regenerated files are newer
    files = {}
    for p in sorted((rerun / "results").glob("*.json")):
        saved = ROOT / "results" / p.name
        # Only files the rerun regenerated: checks that need extra packages (pyfixest, synthdid, wildboottest) aren't in run_all.py.
        if not saved.exists() or p.stat().st_mtime < since:
            continue
        a, b = numbers(json.loads(saved.read_text(encoding="utf-8"))), numbers(json.loads(p.read_text(encoding="utf-8")))
        differ = sorted(k for k in set(a) | set(b) if k not in a or k not in b or abs(a[k] - b[k]) > TOL)
        files[p.name] = {"numbers": len(a), "differ": len(differ), "examples": differ[:5]}
    out = {"date": dt.date.today().isoformat(), "rerun_commit": commit, "tolerance": TOL, "files": len(files),
           "numbers_compared": sum(f["numbers"] for f in files.values()),
           "numbers_differing": sum(f["differ"] for f in files.values()), "by_file": files}
    print(json.dumps({k: v for k, v in out.items() if k != "by_file"}, indent=1))
    for name, f in files.items():
        print(f"  {name}: {f['numbers']} numbers, {f['differ']} differ")
    if "--save" in sys.argv:
        (ROOT / "results" / "rerun_comparison.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
