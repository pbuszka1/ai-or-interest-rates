"""Build docs/SCORECARD.md: every analysis on one page, numbers read from results/*.json.

Numbers come from the saved results, so the scorecard can't drift from what the code found.
The "when did it change" and "verdict" columns are judgments; they're written out below in
plain text so a reader can see exactly where judgment enters.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def j(name: str) -> dict:
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else {}


def pct(x: float | None, signed: bool = True) -> str:
    return "–" if x is None else (f"{100 * x:+.1f}%" if signed else f"{100 * x:.1f}%")


def ci(c: list | None) -> str:
    return "–" if not c else f"[{100 * c[0]:+.1f}, {100 * c[1]:+.1f}]"


def main() -> None:
    occ_sc, var, ind, qwi, occ, inj = (j("synth_computer.json"), j("synth_variants.json"), j("sdid_industry.json"),
                                       j("sdid_qwi.json"), j("sdid_occupation.json"), j("injection_test.json"))
    es, eo = j("event_study_industry.json"), j("event_study_occupation.json")
    rw = j("review_robustness.json").get("test_a_employment_weighted", {})
    # Smallest effect detectable with 80% power at 5% (one-sided test of a decline):
    # (1.645 + 0.84) x null spread, scaled up because the planted cut starts in 2023, partway
    # through the post period. Read from the injection test's no-effect rows.
    mde = {}
    for r in inj.get("rows", []):
        if r["size"] == 0.10:
            share = r["planted_avg_effect"] / -r["size"]
            mde[r["design"]] = (1.645 + 0.84) * r["null_sd"] / share
    chosen = next((v for v in var.get("variants", []) if v["variant"] == var.get("chosen")), {})

    rows = [
        {"name": "1. Synthetic control, computer **occupations**", "data": "CPS, 1 occupation group vs 36 low-exposure groups",
         "outcome": "log employment", "est": pct(occ_sc["main"]["gap_by_year"].get("2025")) + " (2025 gap)",
         "ci": f"p = {occ_sc['main']['p_value']:.2f}", "fit": f"hold-out error {pct(occ_sc['holdout']['rmspe_2018_21'], False)}",
         "when": "–", "dotcom": "can't fit", "mde": "–",
         "verdict": "**Not usable.** Failed its hold-out check (error 5× the fit error). Kept as a documented negative result."},
        {"name": "2. Synthetic control, computer systems design **industry** (variant E)", "data": f"CES payroll, 1 industry vs {chosen.get('n_donors')} lower-exposure industries",
         "outcome": "log employment", "est": pct(chosen.get("gap_2026")) + " (2026 gap)", "ci": f"p = {chosen.get('p_value')}",
         "fit": f"hold-out error {pct(chosen.get('holdout_error_2018_19'), False)}", "when": "Gap opens 2023–24",
         "dotcom": "can't fit (1990s tech outgrew every donor)", "mde": "≈16–20% (placebo spread)",
         "verdict": "**Inconclusive.** Timing fits AI, but the gap is inside the range of placebo industries."},
        {"name": "3. Synthetic DiD, high vs low exposure **industries**", "data": f"CES payroll, {ind['2022_high_pre_ends_2019']['n_treated']} vs {ind['2022_high_pre_ends_2019']['n_controls']} industries",
         "outcome": "log employment", "est": pct(ind["2022_high_pre_ends_2019"]["tau"]), "ci": ci(ind["2022_high_pre_ends_2019"].get("ci95")),
         "fit": f"pre-path RMS {pct(ind['2022_high_pre_ends_2019']['pre_path_rms'], False)}",
         "when": f"{pct(ind['2022_high_pre_ends_2019']['by_year']['2022'])} above normal by 2022 (pandemic boom), back to {pct(ind['2022_high_pre_ends_2019']['by_year']['2026'])} by 2026",
         "dotcom": f"{pct(ind['dotcom_high']['tau'])} (no AI; see caveats in the paper)", "mde": mde.get("ces"),
         "verdict": f"**No AI-specific decline.** The late-2021-baseline estimate ({pct(ind['2022_high']['tau'])}, significant) is the pandemic boom unwinding."},
        {"name": "4. Synthetic DiD, **hires by age** in high vs low exposure industries", "data": "QWI hires, 40 vs 43 industry × region units",
         "outcome": "log hires 22–24 minus 35–44", "est": pct(qwi["primary_2019_baseline"]["gap"]["tau"]),
         "ci": ci(qwi["primary_2019_baseline"]["gap"].get("ci95")), "fit": f"pre-path RMS {pct(qwi['primary_2019_baseline']['gap']['pre_path_rms'], False)}",
         "when": "Opened 2020–21 (−5.9, −9.7 in the version that keeps the pandemic), flat 2022–25",
         "dotcom": f"{pct(qwi['dotcom']['gap']['tau'])} (no clear tilt; wide interval)", "mde": mde.get("qwi"),
         "verdict": "**Real early-career squeeze, but pandemic-timed.** It predates ChatGPT and hasn't widened since."},
        {"name": "5. Synthetic DiD, **young vs old in exposed occupations**", "data": "CPS, 19 vs 26 occupation groups",
         "outcome": "log employment 22–27 minus 35–65", "est": pct(occ["primary_2019_baseline"]["high"]["gap"]["tau"]),
         "ci": ci(occ["primary_2019_baseline"]["high"]["gap"].get("ci95")),
         "fit": f"pre-path RMS {pct(occ['primary_2019_baseline']['high']['gap']['pre_path_rms'], False)}",
         "when": "In the version that keeps the pandemic: opened 2021 (−9), about −10 to −14 through 2025, −23 in 2026 (Jan–Sep)",
         "dotcom": f"{pct(occ['dotcom']['high']['gap']['tau'])} (no clear tilt; wide interval)", "mde": mde.get("cps"),
         "verdict": "**Borderline, mostly pre-ChatGPT, with a possible AI-timed widening in 2026.** The 2026 move matches Stanford's direction but is one partial year of noisy data."},
    ]
    if es:
        p, q = es["primary"], es["primary"]["delta_top_vs_bottom_quartile"]
        rows.append(
            {"name": "6. **Round 2, Test A (pre-registered):** dose-response event study, all industries",
             "data": f"CES payroll, {p['n_industries']} industries, continuous exposure",
             "outcome": "log employment; Δ = mean(2024–26) − 2022, top vs bottom exposure quartile",
             "est": pct(q["est"]), "ci": f"{ci(q['ci95'])}; wild bootstrap p = {p['wild_bootstrap_p']:.2f}",
             "fit": f"pre-trend F-test p = {p['pretrend']['p']:.3f} (**fails** the pre-set p > 0.10 rule)",
             "when": f"Exposed industries {pct(p['path']['2022']['theta'])}/SD above 2019 by 2022, sliding back from 2023, {pct(p['path']['2026']['theta'])}/SD in 2026",
             "dotcom": f"{pct(es['dotcom']['delta_top_vs_bottom_quartile']['est'])} (broadly similar shape, but this placebo fails its own pre-trend test)",
             "mde": es["ramp_injection"]["mde_80pct_analytic"],
             "verdict": "**Failed design, not interpreted.** The estimate would clear the Holm threshold, but the pre-trend check fails, so the pre-registered rules report it as failed. "
                        "Across six versions of the industry score the pre-trend p runs from 0.067 to 0.107 while the estimate stays between −3.7% and −5.0%. "
                        f"About 40% goes with the boom control ({pct(es['boom_controlled']['delta_top_vs_bottom_quartile']['est'])}), "
                        "and the dot-com bust looked broadly similar, although that placebo fails its own pre-trend test. "
                        f"Employment-weighted (`results/review_robustness.json`): {pct(rw['est'])}, {pct(rw['boom_controlled']['est'])} with the boom control, "
                        f"and {pct(rw['dotcom']['est'])} in the dot-com bust."})
    if eo:
        p, q, a = eo["primary"], eo["primary"]["delta_top_vs_bottom_quartile"], eo["age"]
        qa = a["delta_young_minus_old_top_vs_bottom_quartile"]
        rows += [
            {"name": "7. **Round 2, Test B (pre-registered):** occupations *within the same industry*",
             "data": f"CPS, {p['n_cells']} occupation × sector cells ({p['n_groups']} groups, {p['n_sectors']} sectors), annual",
             "outcome": "log employment; Δ = mean(2024–26) − 2022, top vs bottom exposure quartile",
             "est": pct(q["est"]), "ci": f"{ci(q['ci95'])}; P(below 0) = {p['prob_below_zero']:.0%}",
             "fit": f"pre-trend p = {p['pretrend']['p']:.2f} (passes)",
             "when": "Exposed occupations gained slightly within industries 2021–23, flat since",
             "dotcom": "n/a (2003 coding break); fake-2016 placebo " + pct(eo["fake_2016"]["delta_top_vs_bottom_quartile"]["est"]),
             "mde": "≈9% on shuffled exposure; ≈13% using the actual standard error",
             "verdict": "**Bounded null.** Inside industries, AI-exposed occupations haven't lost ground since ChatGPT; "
                        "rules out shifts beyond about ±6%. Robust to dropping computer jobs and tech-heavy sectors."},
            {"name": "8. Test B age version (secondary): young vs older workers in the same occupation × industry cells",
             "data": f"CPS, {a['n_cells']} cell × age units ({a['n_young_cells']} young, {a['n_cells'] - a['n_young_cells']} older)", "outcome": "log employment, 22–27 minus 35–65",
             "est": pct(qa["est"]), "ci": f"{ci(qa['ci95'])}; P(below 0) = {a['prob_below_zero']:.0%}",
             "fit": f"pre-trend p = {a['pretrend']['p']:.2f} (passes)",
             "when": "Young share in exposed cells rose 2011–18, fell from 2022, sharpest in 2026 (−7.1%/SD)",
             "dotcom": "n/a", "mde": "≈20% (Round 1 CPS age design)",
             "verdict": "**Same direction as Stanford, not conclusive.** Mostly the 2026 drop, in one partial year of the same "
                        "CPS data as row 5; needs the ACS."}]
    ces10 = next((r["power"] for r in inj.get("rows", []) if r["design"] == "ces" and r["size"] == 0.10), float("nan"))
    fmt_mde = lambda m: m if isinstance(m, str) else ("pending" if m is None else f"≈{100 * m:.0f}%")

    lines = ["# Results Scorecard", "",
             "Every analysis in one place. Numbers are generated from `results/*.json` by "
             "`scripts/build_scorecard.py`; the *when* and *verdict* columns are judgments, written "
             "out in that script. Estimates use the pre-pandemic (2019) baseline unless noted. "
             "Every correction, with its date, is in the README.", "",
             "| Analysis | Data | Outcome | Estimate | 95% CI / p | Fit quality | When did it change? | Same test in the dot-com bust | Smallest detectable effect (80% power) | Verdict |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['name']} | {r['data']} | {r['outcome']} | {r['est']} | {r['ci']} | {r['fit']} | "
                     f"{r['when']} | {r['dotcom']} | {fmt_mde(r['mde'])} | {r['verdict']} |")
    lines += ["", "## Bottom line so far", "",
              "- **Pre-registered verdict:** Verdict 4, can't tell yet. Test B passes and shows no change; Test A fails "
              "its pre-trend check, and Verdict 3 needs both to pass. The label rests on a pre-trend test near its "
              "threshold (p from 0.067 to 0.107 across versions of the industry score), so the estimates are the better guide.",
              f"- **Industry employment:** a {pct(es['primary']['delta_top_vs_bottom_quartile']['est'])} slide since 2022 in AI-exposed industries, "
              "about 40% of it the pandemic-era boom unwinding. The design fails its pre-trend check. The "
              "dot-com comparison is suggestive only.",
              "- **Early-career hiring:** a real tilt away from young workers in exposed industries "
              "and occupations, but it started in 2020–21, before ChatGPT. A stable gap can't "
              "rule out AI preventing a recovery.",
              "- **The one AI-consistent signal:** young workers' share in the most exposed "
              "occupations widened further in 2026. One partial year of noisy survey data, and the "
              "estimate ranges from about -1% to -11% across specifications, -0.9% when young and older workers are compared within the same cells "
              "(`results/young_worker_robustness.json`), so worth watching, not concluding.",
              "- **What the designs can and can't see** (injection tests, `results/injection_test.json`): "
              f"the industry design detects a 10% AI-timed cut {ces10:.0%} of the time and dates it correctly."
              " The two age designs (QWI, CPS) can't reliably detect cuts under about 20–24%, so "
              "\"no post-ChatGPT widening\" there is weak evidence either way. That's where more data "
              "(the ACS) would help.", "",
              "## How the injection tests work", "",
              "Random control units (no AI-exposure shock) are relabelled as treated, a known cut is "
              "planted in their outcome from 2023-Q1, and the exact pipeline is rerun, 100 times per "
              "size. That measures bias, power, false alarms and whether the method dates the change "
              "correctly, using the real data's noise. Each draw uses half the control pool, so the real "
              "designs (with more units) are somewhat more powerful than these numbers, especially QWI.", ""]
    (ROOT / "docs" / "SCORECARD.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote docs/SCORECARD.md")


if __name__ == "__main__":
    main()
