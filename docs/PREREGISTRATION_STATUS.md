# Pre-Registration Status

Each commitment in [PREREGISTRATION.md](PREREGISTRATION.md), checked against the code on 2026-11-02 and updated 2026-11-03. The pre-registration itself
is unedited. Its SHA-256 hash at freezing was `F72F3BACEB01C33C2B99707B125D2D6E767821E148F78CF939ED9A88E16561FF`, recorded in my own project log
(kept in my private working copy) rather than independently timestamped, and this repository's history starts after the
file was written, so git alone can't prove when it was frozen. Anyone can check that the file still matches that hash.

**Status:** Done = as written. Changed = differs from the plan, with the effect shown. Not run = disclosed and no results held back.

## Shared definitions

| Commitment | Status | Note |
|---|---|---|
| Exposure: average of z-scored GPT-4 and human beta ratings | Done | The average is re-standardized so one unit is one SD; logged before the estimates |
| Industry scores use a 2015–19 occupation mix | Done, after two corrections | First run used 2020–25: Test A −3.7%, pre-trend p = 0.078. The registered mix gave −4.1%, p = 0.107. Fixing two industry-matching errors then gave −5.0%, p = 0.073. Each step changed the verdict |
| Robustness: each rating alone, AIOE, actual-use scores | Not run | Part of the specification curves |
| Primary outcome: log employment | Done | |
| Secondary outcomes: hours, output, earnings | Not run | |
| Reference 2019; drop 2020-Q1 to 2021-Q2 | Done | Test B is annual, so it drops 2020 and the first half of 2021 |
| Clustered errors, wild bootstrap (999), randomization (1,000), two-sided 5% tests | Done | For both primaries |
| Holm correction across Tests A and B | Done | |
| Ramp injection before reading results | Changed | Effects are planted on shuffled exposure across real units, not in "units known to have no exposure shock." Logged before the estimates. Test B's power is overstated on this measure: about 9% shuffled, about 13% with the actual standard error |
| Every design runs a ramp injection | Changed | The triple difference has none; its roughly 13% threshold is analytic only |

## Test A

| Commitment | Status | Note |
|---|---|---|
| All 99 CES industries | Changed | 94 are observed in every quarter of the window. Those 94 cover about 86% of private payrolls |
| Equation, unweighted, clustered by industry | Done | |
| Employment-weighted robustness check | Done late | First omitted. −5.5% [−9.0, −2.1], pre-trend p = 0.104. With boom control −3.1%. Dot-com placebo −4.0%, which fails its pre-trend test (p = 0.027) |
| Headline: mean(θ 2024+) − θ2022 | Done | |
| Pre-trend 2015–18, p > 0.10 | Done | p = 0.073, a fail (0.067 under `pyfixest`'s small-sample convention) |
| Boom control, dot-com placebo, quintiles | Done | |
| Fake date 2017 | Done | −0.1%; now also reported in the paper |

## Test B

| Commitment | Status | Note |
|---|---|---|
| Cells, sector × year effects, weights, clustering, 2011–2026 | Done | 108 of 109 groups and 14 sectors after the filter |
| Cells need 30 respondent-months | Done | Applied in every year; the filter through 2019 is reported as a check |
| Equation includes occupational cyclicality | Done | 3 small groups (0.08% of employment) have no estimate and are set to 0 |
| Pre-trend 2012–18, fake 2016, drop computer jobs, drop information and professional | Done | "Information and professional" is carved out of IND1990; logged |
| Occupation-specific linear trend | Changed | Run per cell. The plan's version gives −2.0%, the cell version −1.9% |
| Age version | Changed | Omits the cyclicality term. Adding it gives −6.5%. A bug first left out the sector-by-age-by-year effects; corrected, the estimate is −6.3%, p = 0.18 |
| ACS rerun | Not run | |

## Other tests and decision rules

| Commitment | Status | Note |
|---|---|---|
| Test C (actual use), D (productivity), E (Section 174), F (earnings and hours) | Not run | Stated in the paper |
| Specification curves | Not run | Stated in the paper |
| Verdict rules | Done | Verdict 4 as written: Test A fails its pre-trend check. Across six versions of the industry score, that p-value runs from 0.067 to 0.107, so the label is fragile |
| Log every change as a deviation and report the original | Done after audit | This audit added the items above that were missing |
