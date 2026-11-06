# Key Choices and Why

Every choice that shaped the results, on one page: what was chosen, why, and what I could have
done instead. Useful for answering "why did you do it that way?"

## The question

| # | Decision | Why | Alternative not taken |
|---|---|---|---|
| 1 | Ask "AI or interest rates (plus over-hiring and Section 174)?" | The popular story credits AI, but rate hikes, the end of the pandemic boom and a tax change all hit tech in 2022 too | Assume AI and measure its size |
| 2 | Reach back to the 1990s and use the dot-com bust as a comparison | A tech bust with no AI shows what "exposed industries fall in a downturn" looks like on its own | Start in 2015 |
| 3 | Frame the public write-up as a first attempt and invite feedback | It's honest (the verdict is "can't tell yet"), and the narrow claims hold up either way | Present it as a definitive answer |

## Data

| # | Decision | Why | Alternative not taken |
|---|---|---|---|
| 4 | CPS with all 18–65 year olds, no labor-force filter in the extract | Filtering on labor-force status could add selection bias; employment status is filtered in code instead | Extract only labor-force participants |
| 5 | Census QWI for hires by age | Actual hires from administrative records, not a survey; hiring is how Stanford found the effect working | CPS only |
| 6 | CES payroll data for industries | Large samples, monthly back to 1990; the CPS was too noisy for single occupations | CPS occupations only |
| 7 | No Lightcast postings data | Paid; postings come from Indeed (2020+) and hiring from QWI | Seek university access |
| 8 | Eloundou et al. exposure scores, GPT-4 and human ratings averaged | The standard measure (Stanford used it); averaging two noisy raters reduces noise | A single rating |

## Design

| # | Decision | Why | Alternative not taken |
|---|---|---|---|
| 9 | Build a synthetic comparison group from other jobs | Before-vs-after on tech alone mixes AI with everything else that changed | A naive before/after |
| 10 | Move from occupations to industries for synthetic control | The occupation version failed its hold-out check (error 5× the fit error) | Keep tuning the failed fit |
| 11 | Push for a better fit, then pre-set a rule for choosing among variants | The first fit looked "pretty horrible"; a rule chosen before seeing results stops cherry-picking | Pick the variant with the nicest result |
| 12 | Synthetic DiD for the pooled high-vs-low comparison | One treated unit can only detect huge effects (about 16–20%); pooling many units averages out noise | More single-unit synthetic controls |
| 13 | Drop the pandemic quarters (2020-Q1 to 2021-Q2) | COVID hit in-person, low-exposure industries for reasons unrelated to AI or rates | Keep them (reported as robustness) |
| 14 | 2019 as the baseline, not late 2021 | Time weights put 81% on 2021-Q4, the peak of the pandemic boom, which flipped the result | Let the method choose its baseline |
| 15 | Remove seasonality from QWI hires | Time weights locked onto one season; young hiring swings ±15% a year | Use the "seasonally adjusted" series as-is |
| 16 | Exposure as a continuous dose in Round 2 | Uses every industry, and a smooth gradient is harder for a confounder to fake | High vs low groups |
| 17 | Compare occupations *within the same industry* (Test B) | Absorbs everything that hits whole industries: rates, the boom, tax rules | Model each confounder |

## Checks and inference

| # | Decision | Why | Alternative not taken |
|---|---|---|---|
| 18 | Injection tests (plant a known effect, rerun) | A worry that the designs weren't catching the trends well enough; this measures what each design can detect | Trust the standard errors |
| 19 | Pre-register Round 2 before running it | Round 1 made choices after seeing data; fixing them in advance prevents unconscious steering | Keep adjusting along the way |
| 20 | Report Test A as "failed pre-trend check" at p = 0.078 | The rule said p > 0.10; arguing "close enough" after seeing the result is what pre-registration prevents | Call it close enough |
| 21 | Report the probability the effect is below zero | Easier to read than a p-value; the caveat (it says the change is real, not what caused it) is reported with it | p-values only |
| 22 | Three p-values for headlines (clustered, wild bootstrap, randomization) | They rest on different assumptions; agreement means the answer isn't an artifact of one | Clustered SEs only |

## Scope

| # | Decision | Why | Alternative not taken |
|---|---|---|---|
| 23 | Publish with three job datasets and three helpers; leave the remaining pre-registered tests (ACS, productivity, actual use, Section 174, earnings) as next steps | Enough for a clear finding and an honest open question; every extra source makes it harder to explain | Add six more sources first |
| 24 | Measure **potential** AI exposure (what AI *could* do), and treat **actual use** as out of scope | Could-do and actually-used are different questions; measuring real adoption well is a project of its own. Findings are stated as "jobs where AI could do more of the work" | Add usage data (Anthropic Economic Index, Census BTOS) now |

## Changes after seeing results

**2026-11-02: restored the pre-registered 2015–19 occupation mix for industry exposure.** Decision 20 reported Test A as failed
at p = 0.078 on a 2020–25 mix. That mix departed from the plan, so I replaced it. The registered mix passes at p = 0.107, which
changes the verdict from 4 to 3. The alternative was to keep the earlier mix because its verdict was more cautious, but that
would choose a specification for its answer. Both are reported.

**2026-11-03: corrected two industry-matching errors.** Two CES series that combine industries were matched to the wrong
Census industries, and catch-all "not specified" rows were averaged in alongside specific matches. Fixing both gives Test A
−5.0% with a pre-trend p of 0.073, so the verdict goes back from 3 to 4. I kept the fix because the matching was wrong, not
because of where the verdict landed, and I report the estimates ahead of the label: across six versions of the industry
score the pre-trend p runs from 0.067 to 0.107 while the estimate stays between −3.7% and −5.0%.

Decisions 13, 14, 15, the exploratory trend-adjusted estimate, and #21 were made after seeing some
results. Each is logged with its reason, and the original version is reported alongside.
