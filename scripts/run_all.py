"""Run the whole project in order, from raw data to every result, figure and check.

Usage:
  python scripts/run_all.py              # analysis only: needs data/raw and data/interim/cps already in place
  python scripts/run_all.py --download   # also download CES, FRED and QWI (needs FRED_API_KEY and CENSUS_API_KEY in .env)
  python scripts/run_all.py --from sdid_industry   # restart from a step

The CPS extract must be requested from IPUMS and loaded first (docs/IPUMS_EXTRACT.md, then scripts/load_cps.py).
Takes about 90 minutes, mostly the synthetic DiD standard errors and injection tests.
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = ["fetch_fred", "fetch_qwi"]
STEPS = [
    "build_exposure_crosswalk", "build_panel", "estimate_rate_sensitivity", "build_industry_panel", "build_occ_sector_panel",
    "synth_computer", "synth_industry", "synth_variants", "sdid_industry", "sdid_qwi", "sdid_occupation", "injection_test",
    "event_study_industry", "event_study_occupation", "test_b_filter_check", "young_worker_robustness", "review_robustness",
    "validate_against_published", "check_population_controls", "verification_checks", "build_scorecard", "verify_paper_numbers",
    "make_data_manifest",
]
LOGGED = {"synth_computer", "synth_industry", "synth_variants", "sdid_industry", "sdid_qwi", "sdid_occupation", "injection_test",
          "event_study_industry", "event_study_occupation"}


def main() -> None:
    steps = (DOWNLOADS if "--download" in sys.argv else []) + STEPS
    if "--from" in sys.argv:
        steps = steps[steps.index(sys.argv[sys.argv.index("--from") + 1]):]
    for step in steps:
        t0 = time.time()
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / f"{step}.py")], cwd=ROOT, capture_output=True,
                           text=True, encoding="utf-8", errors="replace")
        print(r.stdout, end="", flush=True)
        if step in LOGGED:  # keep the console output beside the results it produced
            (ROOT / "results" / f"{step}_run.txt").write_text(r.stdout, encoding="utf-8")
        print(r.stderr, end="", file=sys.stderr, flush=True)
        print(f"{step}: exit {r.returncode}, {time.time() - t0:.0f}s", flush=True)
        if r.returncode:
            sys.exit(f"stopped at {step}")
    r = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"], cwd=ROOT)
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
