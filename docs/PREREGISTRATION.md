# Pre-Registration: Round 2, "Is AI Affecting Jobs, and Where?"

**Written 2026-10-24, before any of the analyses below has been run.** Every specification,
outcome, comparison group and decision rule is fixed here. Any later change is logged in
[PROJECT_LOG.md](PROJECT_LOG.md) as a **deviation**, with the date and reason, and the
original version is still reported alongside the changed one.

The file's SHA-256 hash at the time of freezing is recorded in the project log entry for
2026-10-24, so anyone can check that this document wasn't edited after the results came in.

---

## Part 0. Why pre-register at all

**The problem it solves.** Round 1 made a dozen reasonable choices *after* seeing data: the
2019 baseline rule, deseasonalizing, dropping pandemic quarters, the median split. Each was
logged and defensible. But every choice made after looking is a fork in the road, and if you
take enough forks you can reach almost any answer without ever lying. Statisticians call
this the **garden of forking paths** (Gelman and Loken, 2013). The cure is to choose the
paths before you can see where they lead.

**What this document does.** It fixes four things in advance:
1. **The questions** and the exact number (the **estimand**) that answers each.
2. **The specifications**: the equations, data, controls, sample windows and inference.
3. **The decision rules**: which results would count as "AI is affecting jobs," which as
   "no effect larger than X," and which as "can't tell."
4. **How power is checked**: each design must show it can see a realistic effect before its
   null result counts as evidence.

**Honesty about what has already been seen.** This is not a pristine pre-registration. We
have already looked at CES industry employment, QWI hires and CPS occupation employment
using synthetic DiD (results in [SCORECARD.md](SCORECARD.md)). What is genuinely new in this
round, and not yet seen in any form:
- the continuous-exposure event study and its dose-response gradient;
- every **within-industry** comparison (occupations inside the same industry);
- every **actual-use** exposure measure (Anthropic Economic Index, Census BTOS);
- **productivity, output and hours** by industry;
- **earnings** outcomes;
- the **ACS** data, which hasn't been downloaded.

Where a test reuses data we've seen (Test A uses CES), the risk is that the specification was
unconsciously shaped by Round 1. The protection is that the specification below follows the
plan written on 2026-10-22 in [METHODOLOGY.md](METHODOLOGY.md) Step 3, before Round 1 ran.

---

## Part 1. What would it take to conclude that AI is affecting jobs?

No single regression can settle a causal question in observational data. What scientists do
instead is called **triangulation** (Lawlor, Tilling and Davey Smith, 2016): run several
analyses whose weaknesses are *different*, and see whether they point the same way. If one
design is fooled by the pandemic boom, another that compares jobs *inside* the same industry
won't be fooled the same way.

Epidemiology formalized a checklist for this when it had to decide whether smoking causes
cancer without running an experiment: the **Bradford Hill criteria** (1965). The ones that
matter here are **temporality** (the cause comes before the effect), **biological gradient**
(more exposure, more effect: dose-response), **specificity** (the effect shows up where the
mechanism predicts and not elsewhere), **consistency** (it appears in different data and
designs), and **coherence** (it fits what we know about how the cause works).

Translated into this project, AI (and not rates, the pandemic or tax changes) would leave
these **fingerprints**:

| # | Fingerprint | What AI predicts | What the alternatives predict | Test |
|---|---|---|---|---|
| 1 | **Timing** | Divergence from 2023, growing as adoption spreads | Rates: from 2022. Pandemic: 2020–21 | All |
| 2 | **Gradient** | Smoothly bigger effects at higher exposure | No clean gradient, or a gradient that was already there before 2022 | A |
| 3 | **Inside industries** | Exposed occupations shrink relative to unexposed ones *in the same industry* | Whole industries move together; no within-industry gap | B |
| 4 | **Actual use** | Effects track where AI is actually used, especially for automation | No link to use | C |
| 5 | **Who** | Young and entry-level workers first, beyond what past busts did to them | All ages, or the same young-worker pattern as 2001 | B (age), ACS |
| 6 | **Mechanism** | Exposed industries' output holds up or rises while hours fall (productivity up) | Rates and busts cut output and hours together | D |
| 7 | **History** | No similar pattern in the dot-com bust | The same gradient in 2001 | A, B placebo |
| 8 | **Robust** | Holds across exposure measures, baselines, data | Only in some specifications | Spec curve |

The decision rules in Part 4 turn this table into a verdict.

---

## Part 2. Shared definitions (used by every test)

### Exposure: the "dose"
- **Primary exposure score** for each unit (industry or occupation group): the average of
  the z-scored GPT-4 rating (`dv_rating_beta`) and the z-scored human rating
  (`human_rating_beta`) from Eloundou et al. (2024). **z-scored** means rescaled to have mean
  0 and standard deviation 1 across units, so "one unit of exposure" means "one standard
  deviation more exposed than average." Averaging the two ratings reduces the noise in either
  one alone (NBER w35110 found LLM-based exposure scores disagree a lot).
  - *Why beta:* beta counts tasks an LLM speeds up directly plus half the tasks it speeds up
    with extra software built on top. It's the authors' own headline measure.
  - *Industries* get the employment-weighted average of their occupations' scores, using
    2015–19 occupation mix (already built in `build_industry_panel.py`).
- **Robustness scores:** each rating alone; Felten, Raj and Seamans' AIOE; and the
  actual-use scores of Test C.

### Outcomes
- **Primary:** log employment (CES payroll for industries; weighted counts for CPS and ACS
  cells).
- **Secondary:** log hours (BLS productivity data; CPS usual hours), log real output and
  output per hour (Test D), log real weekly earnings (CPS `EARNWEEK`, deflated by CPI-U),
  log real new-hire earnings (QWI `EarnHirNS`).

**Why logs:** differences in logs are approximately percentage changes, so a large industry
and a small one are compared on the same scale.

### Time
- **Reference period:** 2019 (all four quarters). Every coefficient is "relative to 2019."
  This repeats the Round 1 rule: the late-2021 baseline was the peak of the pandemic boom, so
  any comparison against it mixes in the boom unwinding.
- **Pandemic quarters dropped:** 2020-Q1 to 2021-Q2, as in Round 1. Reported in the plots as
  a gray band, never estimated.
- **Pre-period for checks:** 2015-Q1 to 2019-Q4.
- **Post-period:** 2022-Q1 onward, read in three pre-set blocks:
  - **2022** ("rates, no ChatGPT"): rate hikes began in March, ChatGPT arrived on
    30 November.
  - **2023** ("early generative AI"): ChatGPT and GPT-4 in use, adoption still low.
  - **2024 to latest** ("adoption"): the period where the Stanford team's estimate first
    becomes significant with firm controls, and BTOS AI use rises.

### Inference (how uncertainty is measured)
- Standard errors **clustered by unit** (industry or occupation group), because one unit's
  quarters aren't independent draws.
- A **wild cluster bootstrap** (Rademacher weights, 999 draws) for the headline numbers,
  because cluster-robust formulas are unreliable with fewer than about 50 clusters, and some
  of our designs are near that.
- **Randomization inference** for the headline numbers: shuffle the exposure scores across
  units 1,000 times, re-estimate each time, and ask how often a shuffled estimate is as
  extreme as the real one. This needs no assumptions about the shape of the errors. It is the
  same logic as the placebo tests in Round 1.
- **Two-sided tests at 5%.** AI could *raise* demand for exposed work (cheaper output means
  more output), so increases count as findings too.

### Multiple testing
There are **two primary hypotheses**: Test A (industries) and Test B (occupations within
industries). Their p-values get a **Holm correction**: the smaller p-value must be below
0.025, the larger below 0.05. The reason: with two shots at significance, the chance of at
least one false alarm is nearly 10%, not 5%; Holm brings it back to 5%. Everything else is
**secondary** and labeled as such. Secondary results can support or undermine the primaries
but can't, on their own, produce a "yes."

### Power before results
Every design runs a **ramp injection test** *before* its real estimate is read: plant a
cut that rises in a straight line from 0% in 2023-Q1 to X% by the last quarter of data, in
units known to have no exposure shock, and rerun the pipeline 200 times for X = 0, 2.5, 5,
10 and 15%. A ramp, not a step, because adoption is gradual (Adversarial Review objection 5).
- **Rule:** if a design's 80%-power detectable effect is above **10%**, its null result is
  reported as "uninformative," not as "no effect." Its point estimate is still reported.

---

## Part 3. The tests

### Test A (primary). Dose-response event study across all industries

**The question:** across all 99 CES industries, does *more* exposure predict a *bigger* fall
in employment after 2023, beyond what was already happening in 2022 and beyond what the
industry's sensitivity to rates and the business cycle predicts?

**Why this test:** Round 1 compared "high" with "low" exposure groups. That throws away
information: an industry at the 74th percentile counted as "middle" and was dropped. Using
exposure as a continuous dose uses every industry, gives more power (Fingerprint 2), and lets
us see whether the effect grows smoothly with exposure, which a confounder is less likely to
mimic.

**The equation:**

  log E(i,t) = α(i) + δ(t) + Σₖ θₖ·X(i)·1[year(t) = k] + Σₖ φₖ·R(i)·1[year(t) = k]
             + Σₖ ψₖ·C(i)·1[year(t) = k] + ε(i,t)

Every symbol:
- **E(i,t):** employment in industry i in quarter t (CES, seasonally adjusted).
- **α(i), industry fixed effect:** each industry's own normal level. It soaks up "hospitals
  are bigger than logging," which isn't an effect of anything.
- **δ(t), quarter fixed effect:** anything that hits every industry in quarter t equally: a
  recession, the general level of interest rates, the pandemic. Because it's absorbed, the
  design is **relative** (Adversarial Review objection 1): it can say "more exposed
  industries did worse," never "AI cut N jobs in total."
- **X(i):** the primary exposure score (z-scored).
- **1[year(t) = k]:** an indicator, 1 if quarter t is in year k and 0 otherwise. One θ per
  year, 2015 to 2026, with 2019 left out as the reference.
- **θₖ, the result:** in year k, how much higher or lower log employment is, relative to
  2019, for an industry one standard deviation more exposed. A θ of −0.02 means about 2% lower
  per standard deviation of exposure.
- **R(i), C(i):** rate sensitivity and cyclicality, estimated from 1990–2019 only (before
  generative AI existed, so AI can't contaminate them), shrunk toward zero by empirical Bayes
  as in Round 1. Their own year-by-year coefficients φ and ψ let rate-sensitive and cyclical
  industries follow their own paths, so a 2022 rate shock that hits them isn't credited to AI.
- **ε:** everything else. Standard errors clustered by industry.
- **Unweighted:** each industry counts once, so the result is about the typical industry,
  matching Round 1. Employment-weighted is a robustness check.

**The headline number:** **Δ_AI = mean(θ for 2024 to latest) − θ₂₀₂₂**.
- *In words:* how much the exposure gradient moved between "rates but no ChatGPT" (2022) and
  "AI adoption" (2024+).
- *Why subtract 2022:* whatever exposed industries were already doing in 2022 can't be AI.
  Subtracting it removes the boom-unwinding level that was already in place. If the unwinding
  continued past 2022, it still shows up in Δ_AI, which is why the boom-controlled version
  and the dot-com comparison below exist.

**Pre-set checks:**
1. **Pre-trend:** θ for 2015–2018 jointly zero (F-test, p > 0.10). If it fails, the design
   is broken and Test A is reported as "failed," not interpreted.
2. **Boom control (secondary version):** add Σₖ κₖ·B(i)·1[year(t)=k], where B(i) is how far
   industry i's employment in 2022-Q2 sat above its own 2015–19 linear trend. This asks:
   among industries that boomed equally, does exposure still predict a decline? It speaks to
   Adversarial Review objection 3. **Caution, stated in advance:** if AI is what stopped the
   boom from lasting, this control removes part of the AI effect. So it isn't the primary;
   the gap between the two versions is reported as "how much of the result runs through the
   boom."
3. **Dot-com placebo:** the same equation on 1996–2003, reference year 1998, with the "2022
   block" mapped to 2001 and the "adoption block" to 2002–03. There was no generative AI, so
   any Δ there is plain tech-cyclicality. Reported as **Δ_AI − Δ_dotcom**, a
   difference-in-differences across two busts.
4. **Fake-date placebo:** pretend ChatGPT arrived in 2017 and compute Δ for 2017–18 vs 2016,
   using 2010–2019 data only.
5. **Shape:** the same regression with exposure in quintile bins instead of a straight line,
   to see whether the gradient is smooth (dose-response) or driven by one group.

**Identifying assumption, in plain words:** without AI, more and less exposed industries
would have kept the same *relative* path after 2022, once their different sensitivity to
rates and the cycle is allowed for. **What would break it:** something else that hit
exposed industries harder starting in 2023–24 (the Section 174 tax rule is the main suspect;
see Test E).

**Power:** CES had a detectable effect of about 6% for the step-shaped synthetic DiD in
Round 1. The ramp injection test here gives the number that counts.

### Test B (primary). Occupations inside the same industry

**The question:** within the same industry, did more exposed occupations shrink relative to
less exposed ones after 2023?

**Why this test:** it is the direct answer to Adversarial Review objection 2 ("industries
aren't jobs"). An accounting firm can replace junior bookkeepers with AI while hiring more
sales staff; the industry total stays flat and Test A sees nothing. More importantly, this
test absorbs **every industry-level shock**, including rates, the pandemic boom, client
budgets and Section 174, because the comparison happens inside each industry in each year.
That's the same logic as the firm-by-time fixed effects that changed the Stanford result,
done with public data.

**Data:**
- **Now:** CPS Basic Monthly (already loaded). Cells are occupation group (109 SOC minor
  groups) × industry sector (`IND1990` grouped into about 13 sectors) × year, 2011–2026.
  Starting in 2011 avoids the 2003 occupation coding break; `IND1990` is IPUMS's
  harmonized industry code, which bridges the Census industry recodes since then.
  Cells need at least 30 respondent-months a year; others are dropped.
- **When downloaded:** ACS 1-year samples via IPUMS USA, 2012–2024 (2025 if released), same
  cells. About 50× the CPS sample, so cells can be finer (occupation group × 3-digit
  industry). 2020 is dropped: the Census Bureau released it only with experimental weights
  because the pandemic wrecked the response rate.

**The equation:**

  log E(o,j,t) = α(o,j) + δ(j,t) + Σₖ θₖ·X(o)·1[t = k] + Σₖ ψₖ·C(o)·1[t = k] + ε

- **o:** occupation group; **j:** industry; **t:** year.
- **α(o,j), cell fixed effect:** e.g. "accountants in manufacturing" have their own normal
  level.
- **δ(j,t), industry-by-year fixed effect:** this is the key term. It absorbs *anything*
  that happened to industry j in year t: rates, the boom, tax rules, demand. What's left is
  only differences between occupations inside the same industry.
- **X(o):** the occupation's primary exposure score.
- **C(o):** occupation cyclicality from the CPS 1990–2019, as in Round 1.
- **θₖ:** in year k, the gap in log employment between occupations one standard deviation
  apart in exposure, *inside the same industry*, relative to 2019. Clustered by occupation
  group.
- **Weights:** each cell weighted by its 2015–19 average employment, so tiny noisy cells
  don't dominate. (This differs from Test A on purpose: CPS cells are noisy in a way CES
  industries aren't.)

**Headline:** Δ_AI = mean(θ for 2024 to latest) − θ₂₀₂₂, as in Test A.

**The age version (Fingerprint 5):** add an age dimension (22–27 vs 35–65) and the term
Σₖ λₖ·X(o)·Young·1[t = k], with cell-by-age and industry-by-age-by-year fixed effects. λ
says whether young workers in exposed occupations fell *further* than older workers in the
same occupations and industries. This is a **triple difference** (exposure × age × time),
the closest public-data analogue of the Stanford design. It's secondary because Round 1's
injection test shows CPS can't see young-worker effects under about 20%; it becomes
informative with the ACS.

**Pre-set checks:** pre-trend 2012–2018 jointly zero; fake date 2016 using 2011–2019; drop
computer occupations (do non-tech exposed jobs show it too?); drop the information and
professional services sectors.

**The main threat, stated in advance:** some exposed occupations (secretaries, clerks) have
been shrinking for decades because of *earlier* computerization. That's a pre-existing
trend, not generative AI. Defence: the pre-trend check, plus a robustness version with an
occupation-specific linear trend fitted on 2011–2019 and projected forward, so θ measures
only the departure from each occupation's own trend.

### Test C (secondary). Actual use instead of potential exposure

**The question:** are the Test A and B effects bigger where AI is *actually* used, not just
where it *could* be?

**Why:** exposure scores say whether an LLM could speed up a task, not whether anyone uses
it (Adversarial Review objection 6). If job changes track actual use, that's specificity
(Fingerprint 4). Noise in the treatment pushes estimates toward zero (**attenuation bias**),
so a better-measured treatment can also reveal an effect the noisy one hid.

**Measures:**
- **Anthropic Economic Index** (Hugging Face, CC-BY-4.0): share of Claude conversations
  mapped to each O*NET task and occupation. Converted to usage per worker by dividing by
  2019 occupation employment, then z-scored. Split into **automation** (Claude does the
  task) and **augmentation** (a person works with Claude). Prediction stated in advance: if
  AI substitutes for labour, automation-heavy use predicts declines and augmentation-heavy
  use predicts smaller declines or growth.
- **Census BTOS** AI use rate by sector, biweekly from September 2023. Used only at sector
  level in Test A. **Caution:** the question wording changed in November 2025 ("producing
  goods or services" became "in any business function"), so only within-wording averages are
  used, and the two periods are never spliced.

**Specification:** the Test A and B equations with the usage score in place of X.

**Built-in weakness, stated in advance:** usage is measured *after* 2023, so it could be
*caused* by employment changes (**reverse causality**), and Claude users aren't a random
sample of workers (heavy in software). These results can support a pattern but can't
establish one on their own.

### Test D (secondary). Productivity: the mechanism test

**The question:** in exposed industries, did output hold up while hours fell?

**Why:** this is the only test that tells AI apart from a downturn by *mechanism* rather
than timing (Fingerprint 6). A rate shock cuts demand, so output and hours fall together. A
labour-saving technology lets the same output be produced with fewer hours, so output per
hour (labour productivity) rises in exposed industries. It also speaks to Adversarial
Review objection 3: an industry whose boom is unwinding loses output; an industry where AI
replaced the boom's workers doesn't.

**Data:** BLS labour productivity for detailed industries (annual, NAICS, real output and
hours worked, from 1987), latest year available, likely 2024 or 2025. Matched to CES
industries by NAICS code; unmatched industries are dropped and the count reported.

**Specification:** the Test A equation, annual, run separately for log real output, log
hours and log output per hour.

**Reading it (fixed now):**

| Output | Hours | Points to |
|---|---|---|
| Flat or up | Down | Labour-saving technology (AI-consistent) |
| Down | Down | Demand shock (rates, boom unwinding) |
| Up | Up or flat | AI raising demand for exposed work, or no effect |

**Weakness:** real output for services (software, consulting) is hard to measure; BLS
partly infers it from inputs, which biases productivity growth toward zero. Annual data
gives only 2–3 post years. Secondary for that reason.

### Test E (secondary). Section 174 as a competing explanation

R&D-heavy industries (share of R&D spending in the NSF BERD survey, top quartile) vs the
rest: if the 2022 tax change mattered, they fall relative to others from 2022 and recover
from 2025, when expensing was restored. Test A is rerun **without** R&D-heavy industries; if
Δ_AI shrinks a lot, Section 174 is a confounder of the AI estimate.

### Test F (secondary). Earnings and hours

The Test B equation with log real weekly earnings and usual hours (CPS), and QWI new-hire
earnings in the Round 1 industry × region units. **No direction predicted:** AI could lower
pay (substitution) or raise it (surviving workers are more productive). Reported as
two-sided descriptive results.

### Specification curve (for Tests A and B)

**Why:** even with a fixed primary specification, a reader should see whether the answer
depends on choices we could reasonably have made differently. A **specification curve**
(Simonsohn, Simmons and Nelson, 2020) runs every combination and plots all the estimates,
sorted, with the choices marked underneath.

Choices varied:
- exposure measure: primary, dv, human, AIOE, usage;
- baseline: 2019 or 2018–19;
- controls: none, rate + cyclical, + boom;
- weights: unweighted or employment-weighted;
- pandemic quarters: dropped or kept;
- sample: all, or computer units dropped.

That's 5 × 2 × 3 × 2 × 2 × 2 = 240 runs per test. Reported: the median estimate, the share of
runs with the same sign as the primary, and the share significant.

---

## Part 4. Decision rules: what each result would mean

Fixed now. "Primary passes" means its pre-trend check passed and the ramp injection test
showed a detectable effect of 10% or less.

**Verdict 1: "Evidence that AI is reducing employment in exposed work."** All of:
1. At least one primary Δ_AI is negative and significant after Holm correction.
2. **Timing:** in that test, θ₂₀₂₂ is not already at least half the size of the 2024+ average
   (otherwise most of it predates ChatGPT).
3. **History:** Δ_AI − Δ_dotcom is negative (Test A), or the Test B placebo in the
   dot-com-era equivalent isn't similar. (Test B can't run on 2001 in the CPS because of the
   2003 coding break, so for B this is replaced by the fake-date placebo passing.)
4. **Robust:** at least 70% of the specification curve has the same sign.
5. **Coherence:** at least two of these four secondary fingerprints agree: actual use (C),
   young workers (B age), productivity (D), and not explained away by Section 174 (E).

**Verdict 2: "Evidence that AI is increasing employment in exposed work."** The same rules
with the sign flipped.

**Verdict 3: "No effect larger than X so far."** Both primaries pass their checks, neither
is significant, and X is the larger of the two 95% confidence interval bounds in absolute
value (expressed per standard deviation of exposure and for the top vs bottom exposure
quartile). This is stated with the date of the latest data, because an effect can appear
later.

**Verdict 4: "Can't tell yet."** Anything else, including a significant result that fails
the timing or history rule. The write-up then says which fingerprint failed and what data
would resolve it.

**Partial verdicts are reported as partial.** For example, "within industries, exposed
occupations declined (B passes), but industry totals didn't (A null)" is a meaningful
finding on its own: reallocation inside industries rather than job loss across them.

---

## Part 5. Order of work

1. Freeze this document; record its hash in the project log.
2. Build the shared exposure score and the ramp injection test (power first).
3. Test A on CES. Then its checks.
4. Test B on CPS. Then download the ACS (walkthrough to be written in
   `docs/ACS_EXTRACT.md`) and rerun B and its age version.
5. Tests D and C, which need new downloads (BLS productivity, Anthropic Economic Index,
   BTOS).
6. Tests E and F; specification curves.
7. Update the scorecard and the adversarial review; apply Part 4.

## References
- Abadie (2021), "Using Synthetic Controls," *Journal of Economic Literature*.
- Brynjolfsson, Chandar and Chen (2025–26), "Canaries in the Coal Mine," Stanford Digital
  Economy Lab.
- Eloundou, Manning, Mishkin and Rock (2024), "GPTs are GPTs," *Science*.
- Felten, Raj and Seamans (2021), "Occupational, industry, and geographic exposure to
  artificial intelligence," *Strategic Management Journal*.
- Gelman and Loken (2013), "The garden of forking paths."
- Handa et al. (2025), "Which Economic Tasks are Performed with AI? Evidence from Millions of
  Claude Conversations" (Anthropic Economic Index).
- Hill (1965), "The Environment and Disease: Association or Causation?"
- Lawlor, Tilling and Davey Smith (2016), "Triangulation in aetiological epidemiology,"
  *International Journal of Epidemiology*.
- Roodman, Nielsen, MacKinnon and Webb (2019), "Fast and wild: Bootstrap inference in Stata
  using boottest" (wild cluster bootstrap).
- Simonsohn, Simmons and Nelson (2020), "Specification curve analysis," *Nature Human
  Behaviour*.
- U.S. Census Bureau, Business Trends and Outlook Survey; BLS Productivity program, detailed
  industries.
