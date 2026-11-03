# Methodology

How this project separates AI from interest rates and the boom-bust cycle. Written before
looking at results, so the tests can't be bent to fit them. If the plan changes, the change
and its reason go in the project log.

*Note added at publication: the project log, the data-source guide and the methods review this plan mentions are
kept in my private working copy. Every change to the plan is summarized in [DECISIONS.md](DECISIONS.md) and the
README's Corrections section.*

## The core problem

We can't "control for everything." What we can do is find comparisons where one cause
changes and the others don't. Three sources of variation do that here:

1. **History.** Tech has had booms, busts and rate-hike cycles before, with no AI: the
   dot-com boom and bust (1995–2003), rate hikes in 1994, 2004–06 and 2015–18, and 2008.
   History tells us what a rate-driven tech bust looks like **without AI**.
2. **Occupations.** Jobs differ in how exposed they are to AI and, separately, in how much
   they swing with rates and the business cycle.
3. **Age.** Within the same occupation, early-career and experienced workers face the same
   rates and the same industry but may be affected differently by AI.

## Outcome data

Full source guide, with access and caveats: the data-source guide. Two
additions since this section was first written: **Census QWI** (actual *hires* by age band
and industry, from administrative records) as a second primary outcome, and a rule to run
every exposure test with **at least three exposure measures**, because LLM-based scores
disagree with each other (NBER w35110).


| Period | Outcome | Source | Why |
|---|---|---|---|
| 1990–2026 | Employment by occupation and age, monthly | IPUMS CPS | The only free source that goes back through the dot-com era with occupation and age; harmonized occupation codes (`OCC2010`) let us compare across decades |
| 1990–2026 | Employment in computer systems design and other industries, monthly | BLS CES via FRED | Large-sample check on the CPS for tech industries |
| 1997–2025 | Employment and wages by detailed occupation, annual | BLS OEWS | Large samples; detailed occupations |
| 2020–2026 | Job postings by category, daily | Indeed Hiring Lab | Postings respond to hiring plans fastest, but only exist from 2020 |

**Known gaps:** postings data before 2020 isn't free (Lightcast, from 2007, is paid), so the
long history uses employment, which reacts more slowly than postings. CPS samples for a
single occupation are small (a few hundred software developers a month), so we pool to
quarters and always report confidence intervals.

## Treatment and control variables

- **AI exposure** by occupation: Eloundou et al. (2024), with Felten et al. AIOE as a
  robustness check. Treated as a fixed trait of the occupation.
- **Rate and cycle sensitivity** by occupation: *estimated by us from 1990–2019*, before any
  generative AI, as the response of each occupation's employment to (a) rate changes and
  (b) the overall business cycle. To avoid "rates rise when the economy is hot," use
  monetary-policy *shocks* (Bauer–Swanson or Romer–Romer series) instead of the raw fed funds
  rate, with the raw rate as a check.
- **Boom size:** how far each occupation's employment rose above its own trend in the run-up
  (1995–2000; 2020–22). Busts tend to undo booms, so this soaks up "over-hiring."

## Analysis plan: exactly what gets estimated

Written 2026-10-22, before any of these models has been run.

### What number are we trying to find?
The project estimates **relative** effects: how much more AI-exposed occupations fell than
less-exposed ones, beyond what their sensitivity to rates, the cycle and the 2020–22 boom
predicts. It cannot say "AI cost the economy N jobs in total," because anything that hits
every occupation at once (a recession, the overall rate level) is absorbed by the
time effects below. That limit goes in every write-up.

### Step 1. Build the analysis table
One row per **occupation group × age group × quarter**, 1990-Q1 to 2026-Q3.
- *Occupation groups:* OCC2010 codes pooled into about 100 groups so each cell has enough
  CPS respondents (detailed codes have a few dozen people a month).
- *Age groups:* 22–27 (early career) and 35–65 (experienced).
- *Outcomes:* log weighted employment (stock); new-hire rate from linked months (flow).
- *Fixed traits of each occupation:* AI exposure (three measures); rate sensitivity and
  cyclicality (Step 2); boom size, meaning 2020–22 growth above its own 2015–19 trend.

The QWI version is the same idea with **state × industry × age × quarter** rows and hires
as the outcome, with state-by-quarter effects added.

### Step 2. Measure each occupation's rate sensitivity *before AI existed*
Using **1990–2019 only**, regress each occupation's quarterly employment growth on current
and lagged monetary policy shocks, and separately on the unemployment rate. The summed
coefficients give two fixed traits: **rate sensitivity** and **cyclicality**. Because they
come from years with no generative AI, AI can't contaminate them.

**Change, 2026-10-22 (after running it):** rate sensitivity could not be estimated from CPS
occupation groups (5% of groups significant, the chance rate, with both Fed surprises and
2-year yield changes). It is now estimated by **industry** from CES payroll data and QWI,
and attached to occupations through their industry mix (shift-share). Cyclicality *is*
estimable from the CPS and is used directly. See the project log for the numbers.

### Step 3. The main event study (Test 2)

  log Y(o,t) = α(o) + δ(t) + Σₖ θₖ·Exposure(o)·1[t = k]
             + Σₖ φₖ·RateSens(o)·1[t = k] + Σₖ ψₖ·Cyclical(o)·1[t = k] + Σₖ κₖ·Boom(o)·1[t = k] + ε

In words:
- **α(o)**, occupation fixed effect: each occupation's normal level (software is bigger than
  cooks; that's not an effect).
- **δ(t)**, quarter fixed effect: anything that hits every occupation in quarter t.
- **θₖ**, the result: in quarter k, how much lower log employment is for an occupation with
  one unit more AI exposure, relative to the 2019 baseline. It's estimated separately for
  every quarter, so we get a curve, not a single number.
- **φ, ψ, κ**: let rate-sensitive, cyclical and boom occupations follow their own paths, so
  their moves aren't credited to AI.
- Standard errors clustered by occupation group.

**Reading the θ curve:**

| θ over time | Points to |
|---|---|
| Flat through 2022, falling from 2023–24 | An AI-linked effect (timing matches ChatGPT and later tools) |
| Falling from early 2022 | Rates or the boom unwinding, not AI (ChatGPT didn't exist yet) |
| Flat throughout | No relative AI effect once controls are in |
| Drifting before 2022 | The design is broken (pre-trend); fix it before believing anything |

### Step 4. The same regression around 2001 (placebo)
Run Step 3 on 1996–2002, before the 2003 coding break, with the same exposure scores and
2000-Q4 as the peak. There was no generative AI, so whatever θ shows there is just "exposed
jobs are tech-cyclical." The AI estimate is the 2022 curve **minus** the 2001 curve at the
same distance from each peak.

### Step 5. Early career (Test 3)
Add Exposure × Young × quarter terms. Their curve answers: do young workers in exposed jobs
fall *more* than older workers in the same jobs? Then compare it to the same curve in 2001,
because young workers lose out in every hiring freeze.

### Step 6. Everything else
Test 1 (no-AI forecast with the dot-com backtest), Test 4 (Section 174 and its 2025
reversal), Test 5 (synthetic control for software) and Test 6 (firms) then check the main
result from different angles. A conclusion needs Steps 3–5 and at least two of these to agree.

### Step 3b. Synthetic control as a co-primary design (Parker, 2026-10-22)
Promoted from a side check (Test 5) to a main analysis alongside the event study. The two
fail in different ways: the event study leans on a linear exposure gradient across all
occupations, while synthetic control builds an explicit stand-in for one group from a
weighted blend of others. If they agree, the result is much stronger.

- **Treated units:** (a) computer occupations as one group (OCC2010 1000–1240); (b) each
  high-exposure occupation group separately (customer service, accounting, writing and
  editing, administrative support, and others in the top exposure third).
- **Donor pool:** occupation groups with **below-median AI exposure** under *both* GPT-4
  and human ratings. Strict version: bottom third only. Donors aren't chosen for looking like
  tech. The blend has to *behave* like tech before treatment, which is the whole point of
  the method.
- **Outcome:** log employment (CPS); hires using the QWI industry version (computer systems
  design vs a blend of low-exposure industries).
- **Pre-period:** 2003-Q1 to 2021-Q4 for the main fit (after the coding break, and including
  the 2008 recession and the 2010s boom). Matching through a crash and a boom is what forces
  the blend to share tech's sensitivity to rates and the cycle.
- **Dot-com validation:** fit on 1990–2000 and look at 2001–02. There was no AI, so the gap
  there shows how large a gap synthetic control produces in a plain tech bust. The 2022+
  gap has to be read against it.
- **When does the gap open?** The fit runs through 2021-Q4, and the gap is plotted from
  2022-Q1. A gap opening with the rate hikes (early 2022) points to rates; one opening after
  ChatGPT (2023+) is consistent with AI.
- **Estimators** (chosen after a review of how other fields build comparison groups):
  synthetic DiD for the headline high- vs low-exposure estimate; classic (and augmented)
  synthetic control for the tech-vs-synthetic-tech chart; generalized synthetic control
  and a Bayesian structural time-series model as robustness checks.
- **Donor matching on rate sensitivity**, not just the employment path, so the 2022 rate
  shock hits the blend about as hard as it hits tech. This is the defence against "history
  bias," a second event at the same time as the treatment, which epidemiologists flag as
  the main threat to this design.
- **Hold-out check:** fit weights on 2003–2017 and confirm they predict 2018–2021 before
  looking at anything after 2021.
- **Inference:** placebo in space (rerun with every donor as "treated") with the post/pre
  RMSPE ratio and its permutation p-value; placebo in time (fake treatment in 2019).
- **Dose-response:** plot each treated occupation's post-2022 gap against its AI exposure.
  If AI is the cause, more-exposed occupations should show bigger gaps, under both
  exposure ratings.

**First run, 2026-10-22: failed the hold-out check.** Classic synthetic control on CPS
occupation groups couldn't build a credible synthetic tech (hold-out error five times the
fit error; permutation p = 0.57). Computer occupations grew faster than all but three small,
noisy low-exposure groups. Next, in order: industry-level synthetic control with CES payroll
data; synthetic DiD and augmented SC on the occupation panel; annual CPS data. Full numbers
in the project log.

**Second run, 2026-10-23: industry level with CES payroll data.** Fit and hold-out
much better (hold-out degradation typical of donors), and computer systems design ends
12–16% below its synthetic counterpart by 2026. But p = 0.29, and leave-one-out shows the
timing depends on one donor (home health care), so it doesn't separate AI from rates. Two
rules added: the pandemic quarters (2020-Q1 to 2021-Q2) are left out of every fit, with a
pandemic-included version as robustness; and every synthetic-control result reports
leave-one-out and a placebo hold-out benchmark.

**Synthetic DiD, 2026-10-23** *(numbers as of that date; current results are in [SCORECARD.md](SCORECARD.md))*. Pooled 21 high-exposure industries against 44 low-exposure
ones. With the planned window the effect is −2.4% (significant), but the time weights put 82%
on 2021-Q4, the post-pandemic peak. With a 2019 baseline it's +1.3% (not significant). **New
rule:** every estimate is reported with both a late-2021 and a pre-pandemic (2019) baseline,
and the pre-pandemic one is primary for the AI question. Time weights are always reported.

**QWI hires by age, 2026-10-23.** Synthetic DiD on 27 industries × 4 regions. Two changes made
after the first run, both logged: inference by a jackknife clustered by industry (placebo
inference was impossible with 48 treated vs 47 controls), and deseasonalizing each unit by its
pre-period quarter-of-year means (time weights had locked onto one season). Rules from now on:
deseasonalize quarterly series with age detail; report the event-time path with any estimate,
because timing is what separates the pandemic, rates and AI.

**Round 2 pre-registration, 2026-10-24.** The next round (continuous-exposure event study,
occupations within industries, actual-use exposure, productivity, earnings, ACS) is fixed in
[PREREGISTRATION.md](PREREGISTRATION.md), including the decision rules for "AI is affecting
jobs," "no effect larger than X" and "can't tell." Where it differs from Steps 3–5 above, the
pre-registration governs: annual rather than quarterly event-time coefficients, a 2019
reference period, the headline "2024+ minus 2022" contrast, and a within-industry design
with industry-by-year fixed effects.

### Robustness, decided now
Every main result is rerun with: each of the three exposure measures; employment vs hires;
CPS vs QWI; including and excluding computer occupations themselves; quarterly vs annual
data; and a placebo ChatGPT date of 2021-Q1.

## The tests, in order

### Test 1. Can a no-AI model predict the 2022–26 bust? (counterfactual forecast)
Fit a model of occupational employment on rate shocks, the business cycle and boom size
using **1990–2019 only**. Feed it the actual 2022–26 rates and cycle. It produces the path
each occupation "should" have followed with no AI. The gap between actual and predicted is
what rates and the cycle **can't** explain.

**Validation first (backtest):** fit the same model on 1990–1999 only and check it predicts
the 2001–03 dot-com bust. If it can't predict a bust we know had nothing to do with AI, we
can't trust its 2022 forecast either.

**AI signal:** the unexplained gap after 2022 should grow with AI exposure, and should start
after ChatGPT (late 2022 to 2024), not at the rate peak (early 2022).

### Test 2. Exposure event study (difference-in-differences with continuous treatment)
For each quarter, estimate how much more high-exposure occupations moved than low-exposure
ones, relative to 2019, controlling for estimated rate sensitivity × rate shocks and boom
size × time. Plot the coefficients over time: flat before late 2022 and then diverging is
the pattern AI would produce.

**The biggest threat, and the placebo that checks it:** AI-exposed jobs (software, analysts,
office work) are also the classic white-collar, tech-cyclical jobs. So run the *same*
regression around the 2001 bust. If high-AI-exposure occupations also fell more in 2001,
then "exposure" is partly picking up tech cyclicality, and the 2022 gap has to be measured
relative to the 2001 one: a **difference-in-differences across the two busts**.

### Test 3. Young vs experienced (triple difference)
Within each occupation, compare workers aged 22–27 with those 35+. Rate shocks and hiring
freezes hit new entrants harder **anyway**, so a young-worker drop alone proves nothing.
Placebo: measure the young-vs-old gap in the 2001 and 2008 busts. The AI signal is a young
gap in exposed occupations **bigger than in past busts**, and bigger than in unexposed
occupations.

### Test 4. Section 174 (2022 start, 2025 reversal)
Compare R&D-heavy occupations (software, engineering, scientists) with similar non-R&D ones
around both dates. If the tax change mattered, R&D roles should fall relative to others from
2022 and recover relative to them from 2025. The reversal makes this the cleanest test in
the project.

### Test 5. Synthetic control for software
Build a weighted mix of other occupations that tracks software employment over 1990–2019,
**including through the dot-com bust**. After 2022, the gap between software and its
synthetic twin estimates the combined shock. Placebo: rerun it with each other occupation as
"treated." Software's gap has to stand out from that placebo distribution to count.

### Test 6. Firms: over-hiring, cuts and "AI-washing" (optional)
Firm-level headcount from SEC filings, 2019–2025. Did the firms that grew headcount most in
2020–22 cut most afterwards? Are layoffs that firms *attribute to AI* (Challenger) concentrated
in those same firms? If so, that's consistent with AI being used as a story for unwinding
over-hiring. This tests the CEO-incentive part of the hypothesis directly.

## What would change our mind

| If we see | It points to |
|---|---|
| Test 1's no-AI model predicts 2022–26 well, and the residual doesn't grow with exposure | Macro and boom-bust, not AI |
| A residual that grows with exposure and starts in 2024, bigger than the 2001 exposure gradient | An AI effect on top of the macro shock |
| An extra young-worker gap in exposed jobs beyond past busts | An AI effect concentrated on early-career hiring |
| R&D roles fall from 2022 and recover from 2025 relative to non-R&D | Section 174 mattered |

## Reporting rules
- Every estimate has a confidence interval; standard errors clustered by occupation.
- All specifications are reported, including ones that disagree.
- Postings and employment results are reported separately, never mixed.
