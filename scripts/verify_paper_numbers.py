"""Every number in the paper, checked against the saved results.

Each check rounds the saved value the way the paper prints it and requires an exact match, so a value that
drifts by more than display rounding fails. Run directly for a table, or through pytest (tests/test_paper_numbers.py).
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def j(name: str) -> dict:
    return json.loads((RES / name).read_text(encoding="utf-8"))


def checks() -> list[tuple[str, float, float, int]]:
    """(label, saved value already scaled as printed, number printed in the paper, decimals printed)."""
    a, b, s, q, o = j("event_study_industry.json"), j("event_study_occupation.json"), j("sdid_industry.json"), j("sdid_qwi.json"), j("sdid_occupation.json")
    v, y, r, f, inj = j("synth_variants.json"), j("young_worker_robustness.json"), j("review_robustness.json"), j("test_b_filter_check.json"), j("injection_test.json")
    c, x = j("synth_computer.json"), j("reestimate_pyfixest.json")
    pa, pb, ag = a["primary"], b["primary"], b["age"]
    A, B, Y = pa["delta_top_vs_bottom_quartile"], pb["delta_top_vs_bottom_quartile"], ag["delta_young_minus_old_top_vs_bottom_quartile"]
    e = next(z for z in v["variants"] if z["variant"] == v["chosen"])
    w = r["test_a_employment_weighted"]
    yy, tb = y["young_vs_older"], y["test_b"]
    out = []

    def est(label, d, value, lo, hi, p=None, pd=2):
        out.extend([(f"{label}: estimate", 100 * d["est"], value, 1), (f"{label}: CI low", 100 * d["ci95"][0], lo, 1), (f"{label}: CI high", 100 * d["ci95"][1], hi, 1)])
        if p is not None:
            out.append((f"{label}: p", d["p"], p, pd))

    # Abstract, Table 5 and Section 6
    est("Test B", B, 0.3, -5.6, 6.1, 0.93)
    out += [("Test B: pre-trend p", pb["pretrend"]["p"], 0.23, 2), ("Test B: pre-trend F", pb["pretrend"]["F"], 1.36, 2),
            ("Test B: wild p", pb["wild_bootstrap_p"], 0.93, 2), ("Test B: randomization p", pb["randomization_p"], 0.91, 2),
            ("Test B: P(below 0) flat", 100 * pb["prob_below_zero"], 47, 0), ("Test B: P(below 0) skeptical", 100 * pb["prob_below_zero_skeptical"], 48, 0),
            ("Test B: cells", pb["n_cells"], 576, 0), ("Test B: groups", pb["n_groups"], 108, 0), ("Test B: sectors", pb["n_sectors"], 14, 0),
            ("Test B: quartile gap", pb["quartile_gap_sd"], 2.55, 2)]
    est("Test A", A, -5.0, -9.0, -0.9, 0.017, 3)
    out += [("Test A: pre-trend p", pa["pretrend"]["p"], 0.073, 3), ("Test A: pre-trend F", pa["pretrend"]["F"], 2.22, 2),
            ("Test A: wild p", pa["wild_bootstrap_p"], 0.016, 3), ("Test A: randomization p", pa["randomization_p"], 0.010, 3),
            ("Test A: P(below 0) flat", 100 * pa["prob_below_zero"], 99, 0), ("Test A: P(below 0) skeptical", 100 * pa["prob_below_zero_skeptical"], 97, 0),
            ("Test A: industries", pa["n_industries"], 94, 0), ("Test A: quarters", pa["n_quarters"], 41, 0), ("Test A: quartile gap", pa["quartile_gap_sd"], 2.52, 2),
            ("Test A: theta 2022 per SD", 100 * pa["path"]["2022"]["theta"], 3.0, 1),
            ("Test A: power, ramp to 10%", 100 * a["ramp_injection"]["rows"][3]["power_empirical_null"], 94, 0),
            ("Test A: analytic 80% threshold", 100 * a["ramp_injection"]["mde_80pct_analytic"], 8, 0)]
    for k, val in zip(["Q2", "Q3", "Q4", "Q5"], [1.7, 0.4, -4.0, -3.9]):
        out.append((f"Test A quintile {k}", 100 * a["quintiles_delta_vs_Q1"][k]["est"], val, 1))
    est("Test A boom control", a["boom_controlled"]["delta_top_vs_bottom_quartile"], -2.9, -7.1, 1.2)
    out.append(("Test A boom control: pre-trend p", a["boom_controlled"]["pretrend"]["p"], 0.28, 2))
    est("Dot-com placebo", a["dotcom"]["delta_top_vs_bottom_quartile"], -3.9, -6.6, -1.1)
    out += [("Dot-com: pre-trend F", a["dotcom"]["pretrend"]["F"], 8.41, 2), ("Dot-com: pre-trend p", a["dotcom"]["pretrend"]["p"], 0.0004, 4),
            ("Primary minus dot-com", 100 * a["primary_minus_dotcom_quartile"], -1.1, 1)]
    est("Trend-adjusted Test A", a["exploratory_trend_adjusted"]["delta_top_vs_bottom_quartile"], -6.1, -9.5, -2.7)
    est("Test A fake 2017", a["fake_2017"]["delta_top_vs_bottom_quartile"], -0.1, -2.2, 2.0)
    est("Test A weighted", w, -5.5, -9.0, -2.1)
    out += [("Test A weighted: pre-trend p", w["pretrend"]["p"], 0.104, 3), ("Test A weighted: leave-one-out low", 100 * w["leave_one_out_range"][0], -6.4, 1),
            ("Test A weighted: leave-one-out high", 100 * w["leave_one_out_range"][1], -4.6, 1)]
    est("Test A weighted, boom control", w["boom_controlled"], -3.1, -6.6, 0.3)
    est("Test A weighted, dot-com", w["dotcom"], -4.0, -6.3, -1.6)
    out.append(("Test A weighted, dot-com: pre-trend p", w["dotcom"]["pretrend"]["p"], 0.027, 3))
    mixes = r["test_a_other_mixes"]
    out += [("Test A, 2020-25 mix (corrected matching): pre-trend p", mixes["2020_25"]["pretrend"]["p"], 0.067, 3),
            ("Test A, smallest of the six versions' estimates", 100 * min(-0.0367, -0.0414, -0.0386, A["est"], mixes["2020_25"]["estimate"]["est"], mixes["2020_22"]["estimate"]["est"]), -5.0, 1),
            ("Test A, largest of the six versions' estimates", 100 * max(-0.0367, -0.0414, -0.0386, A["est"], mixes["2020_25"]["estimate"]["est"], mixes["2020_22"]["estimate"]["est"]), -3.7, 1),
            ("Test A pre-trend p, largest of the six versions", max(0.078, 0.107, 0.095, pa["pretrend"]["p"], mixes["2020_25"]["pretrend"]["p"], mixes["2020_22"]["pretrend"]["p"]), 0.107, 3)]

    # Round 1
    hi = s["2022_high_pre_ends_2019"]
    out += [("SDiD industries", 100 * hi["tau"], 0.7, 1), ("SDiD industries: CI low", 100 * hi["ci95"][0], -2.9, 1), ("SDiD industries: CI high", 100 * hi["ci95"][1], 4.3, 1),
            ("SDiD industries: pre-path RMS", 100 * hi["pre_path_rms"], 0.6, 1), ("SDiD industries: late-2021 baseline", 100 * s["2022_high"]["tau"], -3.9, 1),
            ("SDiD industries: treated", s["2022_high"]["n_treated"], 20, 0), ("SDiD industries: controls", s["2022_high"]["n_controls"], 43, 0),
            ("SDiD industries: 2022 gap", 100 * hi["by_year"]["2022"], 5.8, 1), ("SDiD industries: 2026 gap", 100 * hi["by_year"]["2026"], -3.9, 1),
            ("SDiD industries: late-2021 time weight on 2021-Q4", 100 * s["2022_high"]["time_weights_over_1pct"]["2021.75"], 81, 0)]
    g = q["primary_2019_baseline"]["gap"]
    out += [("QWI gap", 100 * g["tau"], -9.3, 1), ("QWI gap: CI low", 100 * g["ci95"][0], -18.3, 1), ("QWI gap: CI high", 100 * g["ci95"][1], -0.2, 1),
            ("QWI gap: pre-path RMS", 100 * g["pre_path_rms"], 4.5, 1), ("QWI: pandemic kept", 100 * q["exploratory_pandemic_kept"]["gap"]["tau"], -8.8, 1),
            ("QWI: late-2021 baseline", 100 * q["secondary_late2021_baseline"]["gap"]["tau"], 0.1, 1)]
    for yr_, val in zip(["2022", "2023", "2024", "2025"], [-10.2, -8.6, -8.9, -9.3]):
        out.append((f"QWI gap {yr_}", 100 * g["by_year"][yr_], val, 1))
    qd = q["dotcom"]["gap"]
    out += [("QWI dot-com", 100 * qd["tau"], -2.9, 1), ("QWI dot-com: CI low", 100 * qd["ci95"][0], -9.7, 1), ("QWI dot-com: CI high", 100 * qd["ci95"][1], 3.9, 1),
            ("QWI dot-com: states", q["dotcom"]["n_states"], 20, 0)]
    od, oh = o["dotcom"]["high"]["gap"], o["primary_2019_baseline"]["high"]["gap"]
    out += [("CPS dot-com", 100 * od["tau"], 0.6, 1), ("CPS dot-com: CI low", 100 * od["ci95"][0], -13.1, 1), ("CPS dot-com: CI high", 100 * od["ci95"][1], 14.2, 1),
            ("CPS SDiD by age", 100 * oh["tau"], -12.6, 1), ("CPS SDiD by age: CI low", 100 * oh["ci95"][0], -26.8, 1), ("CPS SDiD by age: CI high", 100 * oh["ci95"][1], 1.6, 1)]
    out += [("Computer systems design 2026", 100 * e["gap_2026"], -13.0, 1), ("Computer systems design: p", e["p_value"], 0.29, 2),
            ("Computer systems design 2022", 100 * e["gap_2022"], 0.2, 1), ("Computer systems design 2024", 100 * e["gap_2024"], -7.2, 1),
            ("Computer systems design without top donor", 100 * e["gap_2026_without_top_donor"], -10.5, 1), ("Computer systems design: top donor weight", 100 * e["max_weight"], 31, 0),
            ("Computer systems design: donors", e["n_donors"], 67, 0), ("Computer systems design: hold-out error", 100 * e["holdout_error_2018_19"], 0.6, 1),
            ("Computer occupations: hold-out error", 100 * c["holdout"]["rmspe_2018_21"], 25.2, 1), ("Computer occupations: donor groups", len(c["donors"]), 36, 0)]
    ces = [z for z in inj["rows"] if z["design"] == "ces" and z["size"] == 0.10][0]
    out += [("Step injection, industries: power at 10%", 100 * ces["power"], 100, 0)]
    for d_ in ["qwi", "cps"]:
        z = [z for z in inj["rows"] if z["design"] == d_ and z["size"] == 0.15][0]
        out.append((f"Step injection, {d_}: power at 15% is about 40%", round(100 * z["power"], -1), 40, 0))

    # Test B, Table 6 and Section 6.2
    for label, key, vals in [("drop computer occupations", "drop_computer_occupations", (-0.4, -6.6, 5.8)), ("drop information and professional", "drop_info_professional", (0.0, -6.2, 6.2)),
                             ("cell trends removed", "own_trend_removed", (-1.9, -8.0, 4.2)), ("fake 2016", "fake_2016", (0.1, -3.6, 3.9))]:
        est(f"Table 6 {label}", b[key]["delta_top_vs_bottom_quartile"], *vals)
    est("Table 6 filter through 2019", f["filter_through_2019"]["all_ages"]["delta_top_vs_bottom_quartile"], -1.0, -7.1, 5.1)
    est("Table 6 college share", r["test_b"]["with_college_by_year"], -0.5, -8.4, 7.4)
    out += [("Test B: slope test p", tb["pretrend_linear_slope_per_year_per_sd"]["p"], 0.27, 2), ("Test B: 2021-22 equal 2019 p", tb["theta_2021_2022_equal_2019"]["p"], 0.50, 2),
            ("Test B: January-September", 100 * tb["january_to_september"]["est"], 0.2, 1), ("Test B: raw rate sensitivity", 100 * tb["with_raw_occupational_rate_sensitivity"]["est"], 0.2, 1),
            ("Exposure and college share correlation", r["test_b"]["exposure_college_correlation"], 0.65, 2),
            ("Test B interval as ramp endpoint, low", 100 * r["test_b"]["interval_as_linear_ramp_endpoint"]["ci95"][0], -8.6, 1),
            ("Test B interval as ramp endpoint, high", 100 * r["test_b"]["interval_as_linear_ramp_endpoint"]["ci95"][1], 9.4, 1),
            ("Test B: standard error", 100 * B["se"], 3.0, 1), ("Test B: shuffled spread", 100 * b["ramp_injection"]["rows"][0]["null_sd"], 2.2, 1),
            ("Test B: 80% threshold on estimand", 2.8 * 100 * B["se"], 8, 0), ("Test B: 80% threshold as ramp", 2.8 * 100 * B["se"] / b["ramp_injection"]["planted_share_of_endpoint"], 13, 0),
            ("Test B: power low end, ramp to 10%", 100 * b["ramp_injection"]["rows"][3]["power_empirical_null"], 88, 0),
            ("Test B: power high end, ramp to 10%", 100 * b["ramp_injection"]["rows"][3]["power_cluster_test"], 90, 0),
            ("Rating correlation, occupations", r["rating_correlation"]["occupations"], 0.89, 2), ("Rating correlation, groups", r["rating_correlation"]["occupation_groups"], 0.91, 2)]
    est("Test B measured from 2019", tb["relative_to_2019"], 4.2, -6.1, 14.6)
    out.append(("Permutation within education groups p", r["test_b"]["permutation_p"]["within_college_quartile"], 0.89, 2))

    # Young workers, Table 7 and Section 6.3
    est("Young vs older", Y, -6.3, -15.5, 2.9, 0.178, 3)
    out += [("Young: pre-trend p", ag["pretrend"]["p"], 0.18, 2), ("Young: P(below 0)", 100 * ag["prob_below_zero"], 91, 0), ("Young: quartile gap", yy["quartile_gap_sd"], 2.53, 2),
            ("Young: young cells", ag["n_young_cells"], 167, 0), ("Young: all cell-age units", ag["n_cells"], 651, 0),
            ("Young: standard error", 100 * Y["se"], 4.6, 1), ("Young: 80% threshold", 2.8 * 100 * Y["se"], 13, 0),
            ("Young: common cyclicality", 100 * r["young_vs_older_with_common_cyclicality"]["est"], -6.5, 1), ("Young: slope test p", yy["pretrend_linear_slope_per_year_per_sd"]["p"], 0.27, 2)]
    for yr_, val in zip(["2022", "2023", "2024", "2025", "2026"], [-1.4, -2.8, -2.2, -2.5, -7.1]):
        out.append((f"Young path {yr_}", 100 * ag["path"][yr_]["theta"], val, 1))
    for key, vals in [("excluding_2026", (-2.3, -11.9, 7.3, 0.64)), ("age_interacted_cyclicality_and_rates", (-4.7, -14.1, 4.6, 0.31)),
                      ("january_to_september", (-2.7, -13.6, 8.2, 0.63)), ("every_year_per_month", (-6.2, -15.0, 2.6, 0.16)),
                      ("filter_through_2019", (-6.1, -14.4, 2.2, 0.15)), ("relative_to_2019", (-9.9, -25.3, 5.5, 0.20)),
                      ("cell_trends_removed", (-11.4, -21.3, -1.4, 0.026))]:
        est(f"Table 7 {key}", yy[key], *vals[:3], vals[3], 3 if vals[3] < 0.05 else 2)
    est("Table 7 college share", r["young_vs_older_with_college_by_age_by_year"], -5.5, -16.2, 5.2, 0.31)
    est("Table 7 same cells only", yy["same_cells_cell_by_year"], -0.9, -10.3, 8.5, 0.85)
    tw = x["two_way_clustering"]
    out += [("Two-way clustered young: CI low", 100 * tw["young_vs_older"]["delta"]["ci95"][0], -13.1, 1), ("Two-way clustered young: CI high", 100 * tw["young_vs_older"]["delta"]["ci95"][1], 0.5, 1),
            ("Two-way clustered young: p", tw["young_vs_older"]["delta"]["p"], 0.065, 3), ("Two-way clustered Test B: CI low", 100 * tw["test_b"]["delta"]["ci95"][0], -7.2, 1),
            ("Two-way clustered Test B: CI high", 100 * tw["test_b"]["delta"]["ci95"][1], 7.7, 1),
            ("Test A boom control removes about 40%", round(100 * (1 - a["boom_controlled"]["delta_top_vs_bottom_quartile"]["est"] / A["est"]), -1), 40, 0)]
    # Section 7, verification
    rc, sr = j("rerun_comparison.json"), j("sdid_reference_check.json")
    out += [("Full rerun: values reproduced", rc["numbers_compared"], 6861, 0), ("Full rerun: values differing", rc["numbers_differing"], 0, 0)]
    for k in ["2022_high_pre_ends_2019", "2022_high"]:
        out.append((f"synthdid unit-weight correlation, {k}", sr[k]["unit_weight_correlation"], 0.999, 3))
        out.append((f"synthdid estimate gap within 0.02 pp, {k}", float(abs(100 * (sr[k]["reference_tau"] - sr[k]["project_tau_saved"])) < 0.02), 1, 0))
    return out


def run() -> list[tuple[str, float, float, bool]]:
    rows = []
    for label, saved, paper, dec in checks():
        shown = round(saved, dec) if dec else round(saved)
        rows.append((label, shown, paper, abs(shown - paper) < 1e-9))
    return rows


if __name__ == "__main__":
    rows = run()
    for label, shown, paper, ok in rows:
        print(f"{'PASS' if ok else 'FAIL'}  {label}: results {shown}, paper {paper}")
    print(f"\n{sum(r[3] for r in rows)} of {len(rows)} pass")
