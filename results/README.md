# Results

Saved output from every analysis, so numbers can be checked without rerunning anything.
`docs/SCORECARD.md` is generated from these files by `scripts/build_scorecard.py`.

| Files | Analysis | Script |
|---|---|---|
| `event_study_occupation.*` | Test B, occupations within sectors, including the age version | `event_study_occupation.py` |
| `event_study_industry.*` | Test A, dose-response across industries | `event_study_industry.py` |
| `sdid_industry.*` | Synthetic DiD, high vs low exposure industries | `sdid_industry.py` |
| `sdid_qwi.*` | Synthetic DiD, hires by age | `sdid_qwi.py` |
| `sdid_occupation.*` | Synthetic DiD, young vs older workers | `sdid_occupation.py` |
| `synth_variants.*` | Synthetic control variants; variant E is the −13% estimate | `synth_variants.py` |
| `synth_industry.*` | Synthetic control, computer systems design (first specification) | `synth_industry.py` |
| `synth_computer*` | Synthetic control, computer occupations (failed its hold-out) | `synth_computer.py` |
| `injection_test.*` | Step injection tests, Round 1 designs | `injection_test.py` |
| `test_b_filter_check.json` | Exploratory: Test B with its cell filter applied through 2019 only | `test_b_filter_check.py` |
| `young_worker_robustness.json` | Exploratory: the young-worker estimate and Test B under alternative specifications | `young_worker_robustness.py` |
| `review_robustness.json` | Employment-weighted Test A (pre-registered check), other occupation mixes, an education control, restricted permutations, and rating correlations | `review_robustness.py` |
| `reestimate_pyfixest.json` | Tests A and B and the young-worker model refit in `pyfixest`, compared with this project's estimates, plus two-way clustering by occupation group and sector | `reestimate_pyfixest.py` |
| `bootstrap_reference_check.json` | Wild cluster bootstrap p-values compared with the `wildboottest` package | `bootstrap_reference_check.py` |
| `sdid_reference_check.json` | Synthetic DiD estimates and weights compared with the `synthdid` package | `sdid_reference_check.py` |
| `verification_checks.json` | Unmatched occupations, code collapses, partial-year divisors, panel structure, quartile gaps | `verification_checks.py` |
| `population_controls_check.json` | Whether January CPS population-control updates move the young-minus-older comparison | `check_population_controls.py` |
| `data_validation.json` | Comparison with published BLS figures | `validate_against_published.py` |
| `rerun_comparison.json` | A full rerun from raw data in a fresh copy, compared number by number with these files | `compare_rerun.py` |

`.json` holds the estimates, `.csv` the tables and series, and `*_run.txt` the console log of the
run (saved by `scripts/run_all.py`). Round 2's ramp injection results are inside the two `event_study_*.json` files.
