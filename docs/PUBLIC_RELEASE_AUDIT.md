# Public Release Audit

Run on 2026-11-03, before making the repository public. It covers the working tree and every object in the git history
as it stood then (2 commits: `4b2e756`, `e674c60`). Safe text and formatting problems were fixed during the audit. The one
history problem was then fixed by replacing the history with a single commit, with the owner's approval; no data file was
deleted.

## Summary

| # | Check | Status | Note |
|---|---|---|---|
| 1 | Secrets, keys, personal paths and artifacts | **PASS** (after fix) | One PDF in the old commit `e674c60` contained a doc-export account ID; that history was replaced |
| 2 | Data redistribution and licensing | **PASS** | No microdata anywhere in history; tracked data is aggregate only |
| 3 | Appendix file map and links | **PASS**, one expected WARN | Every path in the paper's appendix exists. The pre-registration links to a log kept private |
| 4 | Pre-registration hash and data checksums | **PASS** | The hash matches the frozen record. Found and fixed a line-ending bug in the manifest |
| 5 | Stale numbers and text | **PASS** | Current values everywhere; old values appear only as dated history. Byline bug fixed |
| 6 | Notebooks | **PASS** | Sequential execution, no errors, outputs from the current results |
| 7 | Replicator experience | **PASS** | Pinned requirements for every import; README top links both papers and explains how to check |

**Resolved before going public:** the history was replaced with a single fresh commit of the audited tree while the
repository was still private. That removed the account ID in `e674c60` (check 1) and the four documents taken out of the
public copy in this audit, which had still existed in commit `4b2e756`. The old commits no longer resolve on GitHub.

## 1. Secrets, API keys and personal paths

| Item | Result | Status |
|---|---|---|
| API key values | The 4 key values in the 3 `.env` files on this machine were searched for, byte for byte, in every tracked file and every blob in history. None found | PASS |
| Key-shaped strings | No Anthropic or OpenAI keys (`sk-…`), bearer tokens or AWS keys in any blob in history | PASS |
| Local paths and personal details | No `C:\Users`, `/Users/`, `/home/`, local user name or personal email in any text blob in history. Git author email is the GitHub no-reply address | PASS |
| Account ID string (`u_V0XrR7…`) | Not in the working tree. Was present in the text of `paper/ai_or_interest_rates.pdf` in the old commit `e674c60`, from a name chip the doc export printed as a raw ID. The current PDF has the plain byline, and the history was replaced | PASS (fixed) |
| Tracked OS and editor artifacts | None (`.DS_Store`, `__pycache__`, `.ipynb_checkpoints`, `.vscode`, `.idea`, `.claude`, `.env`, `.log`, `.tmp`, `.pyc`), in the tree or in history | PASS |
| `.gitignore` | Covers `.env`, `data/raw`, `data/interim`, `.venv`, `__pycache__` and `.ipynb_checkpoints`. `.DS_Store` and `Thumbs.db` were missing and were added | PASS (fixed) |
| PDF metadata | Title plus the Edge/Skia generator string only; no author, user name or path | PASS |

PDF text was extracted page by page for every PDF in history, because their contents are compressed and a byte search
would miss them.

## 2. Data redistribution and licensing

| Item | Result | Status |
|---|---|---|
| Raw extracts in history | No `.dat`, `.gz`, `.dta`, `.xml`, `.sav` or `.zip` file, and no path under `data/raw/` or `data/interim/`, in any commit | PASS |
| Person-level identifiers | No `CPSID`, `CPSIDP`, `SERIAL`, `PERNUM` or similar column in any tracked CSV or parquet file | PASS |
| `data/processed/` (10 files) | CES-derived tables (`ces_panel.parquet`, `ces_industries.csv`, two `sdid_reference_input_*.csv`); exposure scores by occupation code and group (`exposure_occ2010.csv`, `occ_groups.csv`); rate sensitivity by group (two CSVs); and two CPS tables aggregated to occupation group × age × quarter and occupation group × sector × age × year. In the published copy of the quarterly table, earnings are blanked for cells with fewer than 3 respondents | PASS |
| `results/` | Estimates, series and run logs only | PASS |
| Largest objects in history | 1.69 MB (`panel_occ_age_quarter.parquet`), then the PDFs (under 1 MB each). Nothing near a microdata extract's size (about 1.2 GB) | PASS |
| `LICENSE` | MIT, in the root | PASS |
| Indeed Hiring Lab | No Indeed data file is tracked, only the download script. The README's license table credits it under CC BY 4.0 | PASS |

## 3. Appendix file map and links

Every path in the technical paper's "Appendix: Repository Guide" exists at that path and casing: `docs/SCORECARD.md`,
`docs/PREREGISTRATION.md`, `docs/DECISIONS.md`, `docs/LIMITATIONS.md`, `docs/PREREGISTRATION_STATUS.md`,
`docs/DATA_VALIDATION.md`, `docs/VERIFICATION_AUDIT.md`, `docs/DATA_MANIFEST.csv`, `notebooks/`, `results/`, `figures/`,
`scripts/test_b_filter_check.py`, `scripts/young_worker_robustness.py`, `scripts/review_robustness.py`,
`scripts/reestimate_pyfixest.py`, `scripts/run_all.py`, `scripts/compare_rerun.py` and `results/rerun_comparison.json`.

| Item | Result | Status |
|---|---|---|
| `docs/DATA_SOURCES.md`, `docs/PROJECT_LOG.md` | Removed from the public copy at the owner's request during this audit, along with `docs/LEARNINGS.md` and `docs/METHODS_REVIEW.md`; they remain in the private working copy. Their appendix rows were removed from the paper, and every README and docs link to them was repointed | PASS (by decision) |
| One notebook per analysis | 8 analyses, 8 notebooks (`01` to `08`), matching the README table | PASS |
| README Section 13 | "Corrections" exists, with dated tables for 2026-11-02 and 2026-11-03 | PASS |
| Relative links and images in all `.md` files | All resolve except one: `docs/PREREGISTRATION.md` links to `PROJECT_LOG.md`. The pre-registration can't be edited without breaking its hash; the README explains that its log and review links point to private files | WARN (expected) |

## 4. Pre-registration hash and data checksums

| Item | Result | Status |
|---|---|---|
| `docs/PREREGISTRATION.md` | SHA-256 `F72F3BACEB01C33C2B99707B125D2D6E767821E148F78CF939ED9A88E16561FF`, matching the value frozen in the (private) project log on 2026-10-24. The hash is now published in `docs/PREREGISTRATION_STATUS.md`. It is self-recorded, not independently timestamped, as that file says | PASS |
| `docs/DATA_MANIFEST.csv` against the original data | All 194 raw and processed files match | PASS |
| Manifest against the published processed files | 9 of 10 match. `panel_occ_age_quarter.parquet` differs by design (blanked small-cell earnings), as the README states | PASS |
| Line endings (found and fixed) | Text files were hashed byte for byte, so a CSV checked out by git on Linux (LF) failed against a Windows copy (CRLF). `scripts/make_data_manifest.py` now hashes text files with line endings normalized, and the manifest was regenerated | PASS (fixed) |
| Replicator instructions | The script's docstring and the README's data availability statement explain how to check your own downloads and IPUMS extract with `python scripts/make_data_manifest.py --check` | PASS |

## 5. Stale numbers and text

Current values, checked in the README, scorecard, both papers, docs, scripts and notebook text:

| Result | Current value |
|---|---|
| Test A | −5.0% (95% CI −9.0% to −0.9%); pre-trend F = 2.22, p = 0.073; quartile gap 2.52 SD; p-values 0.017, 0.016, 0.010 |
| Test B | +0.3% (95% CI −5.6% to +6.1%); pre-trend p = 0.23; quartile gap 2.55 SD |
| Young vs older (triple difference) | −6.3% (95% CI −15.5% to +2.9%), p = 0.18 (0.178); pre-trend p = 0.18; SE 4.6%; 651 units (167 young, 484 older); quartile gap 2.53 SD |
| CES synthetic DiD | +0.7% (95% CI −2.9% to +4.3%); 20 high vs 43 low-exposure industries; late-2021 baseline −3.9% |
| QWI synthetic DiD | −9.3% (95% CI −18.3% to −0.2%); pre-period RMS 4.5%; −8.8% with the pandemic quarters kept |
| Verdict | Verdict 4, "can't tell yet" |

Older values (−3.7%, p = 0.078; −4.1%, p = 0.107; −6.8%, p = 0.087; Verdict 3) appear only where the text describes the
history: the README's revision note, its "rule bit" paragraph and Corrections, the dated row in
`PREREGISTRATION_STATUS.md`, the dated Round 1 entry in `METHODOLOGY.md` (now marked as superseded), the history sentence
in notebook 07, and the tests that guard against stale numbers.

Fixed during this audit:

- **The readable paper's byline** printed a raw account ID (`@u_V0XrR7…`) from a name chip. It now reads "Parker Buszka ·
  November 2026" in the source and in `paper/ai_or_interest_rates.pdf`, which was regenerated.
- README: the Stanford figures (13%, 16% and 19% are two different measures, now said so), "82%" to 81%, the dot-com
  placebo years (1996–2003, not 2000–03), the CPS years in the data statement (1990, not 2000), "part of the slide is the
  boom" to "about 40% runs through the boom", the Python version, a duplicated sentence, and future-work wording.
- Notebook 04's conclusion said young hiring "fell about 11% (significant)"; it now gives the current 9% and its interval.
- The scorecard's CPS timing note is now labeled as the version that keeps the pandemic quarters.
- Both papers and the docs no longer point to the four documents kept private.

## 6. Notebooks

| Notebook | Execution counts | Errors | stderr output | Status |
|---|---|---|---|---|
| 01 to 08 | Sequential from 1 in every notebook | 0 | 0 | PASS |

No output contains a local path. The notebooks read the saved results, and all eight were executed after the final
results were produced (03, 04, 07 and 08 again after the figures were redrawn). A full rerun from raw data then
reproduced every saved value (`results/rerun_comparison.json`), so the outputs match `results/`. They were not
re-executed again for this audit.

## 7. Replicator experience

| Item | Result | Status |
|---|---|---|
| `requirements.txt` | 46 pinned packages; covers every third-party import in `scripts/` and `notebooks/` except the three reference packages | PASS |
| Reference packages | `pyfixest` and `wildboottest` need pandas 3, and `synthdid` needs numpy 1.23 and pandas 1.5, so they can't share one environment. Added `requirements-pyfixest.txt` and `requirements-synthdid.txt`, pinned to the versions that produced the saved checks | PASS (fixed) |
| Python version | 3.11 locally, 3.12 in CI; the README now says either works | PASS |
| Top of the README | Links the general-reader paper and the technical version, explains `python -m pytest tests` (no download needed) and `python scripts/run_all.py`, and links the IPUMS extract recipe | PASS (quick start added) |
| Tests, lint and privacy check | 241 tests pass; `ruff check scripts tests` passes; the privacy check passes | PASS |
