# Verification Audit

Checks of the estimators, the data pipeline and every number in the paper, run on 2026-11-03 before public release.
No pre-registered specification was changed. Each check names the script that reproduces it.

**Summary.** The estimators agree with reference packages. Every number in the paper matches the saved results
(`scripts/verify_paper_numbers.py`, all passing after the corrections below). The audit found a typo in a table of an earlier draft, now fixed; a large sensitivity
in the young-worker estimate, now reported; and two industry-matching errors that change Test A's verdict, now corrected.
Every number below is from the corrected run.

| Area | Result |
|---|---|
| Event-study estimates vs `pyfixest` | Match to four decimals in all four models |
| Standard error of the headline contrast | Uses the full clustered covariance matrix |
| Wild cluster bootstrap vs `wildboottest` | Match within simulation error |
| Synthetic DiD vs the `synthdid` package | Estimates within 0.02 percentage points; weights correlate 0.999 |
| Paper numbers vs saved results | All pass after one typo fix (`scripts/verify_paper_numbers.py`) |
| Full rerun from raw data in a fresh copy and environment | 6,861 of 6,861 saved values reproduced (`results/rerun_comparison.json`) |
| Crosswalks, partial years, log conversion | No errors |
| Industry-to-occupation matching | **Two errors found and corrected; verdict changed** |
| Young-worker panel structure | **Large sensitivity found, reported** |

## 1. Estimators

### Event studies (Tests A and B, and the triple difference)

- **Contrast standard error.** `event_study.test` computes `sqrt(c' V c)`, with `V` the full clustered covariance matrix and
  `c` = 1/3 on 2024–26 and −1 on 2022, so the covariance between years is included.
- **Point estimates.** `scripts/reestimate_pyfixest.py` rebuilds each sample from `data/processed` and fits it in `pyfixest`
  without project code. All four models match to four decimals (`results/reestimate_pyfixest.json`).

  | Model | Project | pyfixest | Clusters |
  |---|---|---|---|
  | Test A | −4.9569% | −4.9569% | 94 industries |
  | Test A, employment-weighted | −5.5438% | −5.5438% | 94 |
  | Test B | +0.2522% | +0.2522% | 108 occupation groups |
  | Young vs older | −6.3005% | −6.3005% | 102 occupation groups |

- **Small-sample correction.** Both use `G/(G−1) × (N−1)/(N−K)`. This project counts absorbed fixed effects in K, including
  those nested within clusters; `pyfixest` doesn't count nested ones. Standard errors are therefore 1.3% (Test A), 3.5% (Test B)
  and 3.6% (young vs older) larger here. Test A's pre-trend p is 0.073 here and 0.067 in `pyfixest`: both fail.
- **Pre-trend F-tests** use the same clustered covariance and `F(k, G−1)`.
- **Bug found during this check:** the young-worker model had silently lost its sector-by-age-by-year effects (a pandas index
  mismatch). Fixed in four scripts; `fit()` now refuses missing labels, and `tests/test_fit_labels.py` covers it.

### Wild cluster bootstrap

- **Restricted (WCR).** `event_study.wild_bootstrap_p` imposes the null on the tested contrast, flips each cluster's
  residuals with Rademacher weights, and compares bootstrap t-statistics with the real one. 999 draws, seeded (`default_rng(2026)`).
- **Against `wildboottest`** (`scripts/bootstrap_reference_check.py`, 9,999 draws, single Test A coefficients, since the package
  tests one coefficient at a time):

  | Coefficient | wildboottest p | Project p |
  |---|---|---|
  | 2022 | 0.0079 | 0.0053 |
  | 2024 | 0.3142 | 0.3115 |
  | 2026 | 0.6871 | 0.6869 |
  | 2016 | 0.8658 | 0.8545 |

  The gaps are within the Monte Carlo error at this many draws (about 0.005 near p = 0.3, 0.003 near 0.86, and
  under 0.001 near 0.005, where both agree the coefficient is clearly significant).
- When no bootstrap draw is as extreme as the real statistic, the code reports 0 rather than 1/(B+1). The paper reports such
  values as "below 0.01".

### Synthetic DiD and synthetic control

- **Weights.** Unit and time weights are found by constrained least squares with an intercept (SLSQP), non-negative and summing
  to one (checked). The unit-weight penalty is ζ² × T_pre with ζ = (N_treated × T_post)^¼ × σ, and σ is the standard deviation
  of controls' first differences before treatment, as in Arkhangelsky et al. (2021). The time-weight penalty is 10⁻⁶σ² × N_controls,
  against (10⁻⁶σ)² × N_controls in the reference package. Both are negligible, and the weights match.
- **Against `synthdid` 0.10.1** (a Python port of the reference R package; R isn't installed here). `scripts/verification_checks.py`
  exports the exact matrices and `scripts/sdid_reference_check.py` compares them:

  | Specification | synthdid | Project | Unit-weight correlation | Time weights |
  |---|---|---|---|---|
  | 2019 baseline (reported) | +0.6906% | +0.7064% | 0.999 | All on 2019-Q4 in both |
  | Late-2021 baseline | −3.8971% | −3.8806% | 0.999 | 0.188 on 2019-Q4, 0.812 on 2021-Q4 in both |

  The package misreads fractional quarters as times, so the check passes integer period indexes.
- **Standard errors.** The CES placebo procedure re-estimates both weights on each draw (Algorithm 4). The QWI jackknife keeps the
  full-sample weights and renormalizes the remaining controls (Algorithm 3), by industry cluster. Both follow the paper.
- **Not checked against a package:** the ridge-regularized synthetic control. It's exploratory and reported as inconclusive.

## 2. Data pipeline

### Exposure scores and quartile gaps

- O*NET 8-digit scores are averaged with equal weights within each 6-digit SOC code, as in Eloundou et al. No `drop_duplicates`.
- The primary score z-scores each rating, averages them, and re-standardizes the average. All three steps are unweighted across units.
- The quartile gap is the mean of the top quarter minus the mean of the bottom quarter, unweighted, from one function
  (`event_study.quartile_gap`) used everywhere. Values are 2.523 (Test A), 2.554 (Test B) and 2.534 (young vs older), against
  2.54 for a standard normal distribution.

### Unmatched occupations (1.8% of 2020–25 employment)

All are residual "all other" categories with no SOC score. None is a computer, management or health occupation.

| Census code | Occupation | Share of employment |
|---|---|---|
| 2014 | Social workers, all other | 0.45% |
| 5940 | Office and administrative support workers, all other | 0.41% |
| 4965 | Sales and related workers, all other | 0.24% |
| 2006 | Counselors, all other | 0.16% |
| 9150 | Motor vehicle operators, all other | 0.10% |
| 5165 | Financial clerks, all other | 0.09% |
| 7855 | Food processing workers, all other | 0.09% |
| 4655 | Personal care and service workers, all other | 0.08% |
| 5420 | Information and record clerks, all other | 0.07% |
| 2180 | Legal support workers, all other | 0.05% |

### CPS partial years and the cell filter

- The divisor for cell employment equals the months kept: 12 in full years, 6 in 2021 (July–December), 11 in 2025 (no October) and 9 in 2026.
- The 30 respondent-month filter is applied to raw counts in every year, including the partial ones, and 2021 drops the most cells (659).
  The paper discloses this, and a version applying the filter only through 2019 is reported.

### Log conversion

No script uses `exp()`. Every percentage is 100 × the log-point estimate, as the paper states.

### Raw data and every column the analysis reads

An exploratory notebook (not published, because it shows CPS microdata) prints and profiles every raw file, then checks each column
that feeds a reported result. It runs end to end with no errors.

| Check | Result |
|---|---|
| Raw CPS lines vs loaded rows | 34,569,077 each, equal in all 440 months; only October 2025 is missing |
| Used CPS columns, 2011–2026 | No missing values; every EMPSTAT, OCC2010 and IND1990 code is in the codebook; no employed person has zero weight, an unknown occupation or no industry |
| Test B panel rebuilt from raw rows (2011, 2021, 2026; all three age bands) | Identical: same cells, same respondent counts, employment equal to six decimals; divisors 12, 6 and 9 |
| Test A quarterly table rebuilt from raw series | Identical to the table the analysis uses |
| Sector mapping | Every employed worker gets a sector in every year |
| Occupation-group mapping | Complete from 2020. In 2015–19, 0.43% of employment has OCC2010 codes that exist only before 2020 (largest: gaming supervisors, 181,000 a month; counter attendants, 149,000) and so gets no group |
| Coding breaks | Codes that vanish or jump across 2020 are listed in the notebook, for example Business Operations Specialists, All Other: 258,000 a month in 2019, 555,000 in 2021 |
| CES | No zero or negative values. Two Test A industries (Ship and Boat Building; Other Miscellaneous Retailers) end in August 2026, so their 2026-Q3 average uses two months. Setting September equal to August leaves Test A unchanged (−4.957%) |
| QWI | 45 states report every quarter; 1 of 108 industry-region units is dropped for a zero or missing quarter (motor vehicle manufacturing, Northeast); smallest kept quarterly hires: 30 |
| Monetary policy surprises | All 360 months of 1990–2019 present |
| Exposure scores | Complete for all 448 OCC2010 codes and 109 groups, all within 0 to 1 |

**Effect of the pre-2020 occupation gaps and coding breaks.** Both sit at the 2020 boundary, so they shift some groups' levels between
2019 and 2021. Test B's headline compares 2024–26 with 2022, both after the change, so it is unaffected. Comparisons that reach back to
2019 (Test B measured from 2019, and the CPS synthetic DiD by age) cross the change, and the paper flags them.

## 3. The young-worker panel (Equation 5)

- **Cells.** 167 young cells and 484 older cells (651 series). Every young cell also has an older series, but 317 older cells have no
  young series. The two age groups are not restricted to the same cells.
- **Weights.** Each series is weighted by its own 2015–19 employment in its age band, so the older series carry 85% of the total weight.
- **Cell-by-year effects.** Adding full cell-by-year effects (comparing young and older workers only within the same 167 cells) changes the answer:

  | Version | Estimate | 95% CI | p |
  |---|---|---|---|
  | As written (Eq. 5) | −6.3% | −15.5 to +2.9 | 0.18 |
  | Same cells, cell-by-year effects, young-cell weights | −0.9% | −10.3 to +8.5 | 0.85 |
  | Same cells, cell-by-year effects, equal weights | −4.3% | −15.3 to +6.7 | 0.44 |

  As written, the young gradient comes from 167 cells and the older gradient from 484, so part of the gap reflects which cells are
  compared. This is a specification sensitivity, not a code error. Eq. 5 is the pre-registered version. Reported in the paper
  and `results/young_worker_robustness.json`, using the sector-clustered interval shown here.

## 4. Problems found

| Problem | Effect | Status |
|---|---|---|
| Young-worker model lost its sector-by-age-by-year effects (pandas alignment) | −6.8%, p = 0.087 became −6.3%, p = 0.18 | Fixed and reported |
| An earlier draft's table printed p = 0.32 for the age-interacted controls row; the result is 0.315, which rounds to 0.31 | Typo | Fixed in the paper |
| Two CES aggregate codes (Motor Vehicles and Parts; Beverage, Tobacco and Leather) match no NAICS code, so they get the "Not specified manufacturing" score | Test A pre-trend p falls from 0.107 to 0.091, so the verdict becomes "can't tell yet" | Corrected: mapped to their component NAICS industries |
| "Not specified" catch-all census rows are averaged into every industry in their sector | With both fixes, Test A is −5.0% [−9.0, −0.9], pre-trend p = 0.073 | Corrected: used only when nothing specific matches. Verdict now 4, "can't tell yet" |
| Young-worker estimate depends on whether young and older workers are compared within the same cells | −6.3% vs −0.9% | Reported in the paper and README |
| CPS weekly earnings (EARNWEEK) are 0 from 2023 on | None: no reported result uses earnings | Noted |
| `data/raw/ces/ce.industry` is a saved "Access Denied" page | None: nothing reads it | Noted |
| `docs/IPUMS_EXTRACT.md` says the extract is limited to the labor force; it isn't | None for employment | Corrected in that file |
| `load_cps.py` shrinks integer types chunk by chunk, so UHRSWORKT and DURUNEMP are int8 in 1990–92 and int16 later, and JTYEARS mixes int8 and float | None: no analysis reads these columns, and reading them across years raises an error rather than corrupting values | Noted; the EDA notebook reads them with one schema |

## 5. Review items that didn't hold

- **Duplicate year labels in Figure 1's dot-com panel.** The rendered figure and PDF show each year once; no change needed.
- **The suggested cell counts for Eq. 5** (167 young, 576 older) were wrong: it's 167 and 484.
- **The suggested expected values** (for example −6.8% for young workers) were from an earlier draft. The assertion suite tests the current paper.

## 6. Full rerun

On 2026-11-03 I cloned the published repository into an empty folder, built a new environment from
`requirements.txt`, copied in the raw data, and ran `scripts/load_cps.py` and `scripts/run_all.py` (about 90 minutes).
Every step succeeded and all tests passed. `scripts/compare_rerun.py` then compared every number in the 15 result files the
run regenerates with the saved ones: all 6,861 match to within 10⁻⁹. The one processed file that differs,
`occ_groups.csv`, differs only in floating-point noise (largest difference 3 × 10⁻¹¹ in an employment total).

## 7. Not done

- A rerun on a different computer. The full rerun above used a fresh copy and environment on the same machine.
- A synthdid comparison in R itself; the Python port was used.
- A package comparison for the synthetic control.

## Reproduce

```
python scripts/verification_checks.py          # pipeline checks, SDiD export
python scripts/verify_paper_numbers.py         # 197 paper numbers
python -m pytest tests                         # includes the paper-number checks
# in an environment with pyfixest, wildboottest and synthdid:
python scripts/reestimate_pyfixest.py
python scripts/bootstrap_reference_check.py
python scripts/sdid_reference_check.py
```
