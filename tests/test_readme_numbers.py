"""The README and scorecard headline numbers must match the saved results, and superseded numbers must not come back.

Each expected string is built from results/*.json, so a rerun that moves a number fails this test until the text is updated.
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
SCORECARD = (ROOT / "docs" / "SCORECARD.md").read_text(encoding="utf-8")


def res(name: str) -> dict:
    return json.loads((ROOT / "results" / f"{name}.json").read_text(encoding="utf-8"))


def pct(x: float, signed: bool = True) -> str:
    """Format like the README: unicode minus, one decimal."""
    s = f"{100 * x:+.1f}%" if signed else f"{100 * abs(x):.1f}%"
    return s.replace("-", "−")


ES, EO, RR = res("event_study_industry"), res("event_study_occupation"), res("review_robustness")
SD, QWI = res("sdid_industry")["2022_high_pre_ends_2019"], res("sdid_qwi")["primary_2019_baseline"]["gap"]


def headline_strings() -> list[str]:
    a = ES["primary"]["delta_top_vs_bottom_quartile"]
    b = EO["primary"]["delta_top_vs_bottom_quartile"]
    y = EO["age"]["delta_young_minus_old_top_vs_bottom_quartile"]
    w = RR["test_a_employment_weighted"]
    out = [
        f"{pct(a['est'])}, CI {pct(a['ci95'][0])} to {pct(a['ci95'][1])}",                     # Test A row
        f"pre-trend p = {ES['primary']['pretrend']['p']:.3f}",
        f"**{pct(b['est'])}**, 95% CI {pct(b['ci95'][0])} to {pct(b['ci95'][1])}",             # Test B row
        f"{pct(y['est'])}, CI {pct(y['ci95'][0])} to {pct(y['ci95'][1])}",                     # young workers
        f"Weighted by employment: {pct(w['est'])} (CI {pct(w['ci95'][0])} to {pct(w['ci95'][1])})",
        f"{pct(QWI['tau'])}, CI {pct(QWI['ci95'][0])} to {pct(QWI['ci95'][1])}",               # QWI hires
        f"{pct(SD['tau'])} (CI {pct(SD['ci95'][0])} to {pct(SD['ci95'][1])})",                 # SDiD industries
        f"{SD['n_treated']} high vs {SD['n_controls']} low-exposure industries",
    ]
    return out


@pytest.mark.parametrize("s", headline_strings())
def test_readme_headline_matches_results(s: str) -> None:
    assert s in README, f"README is missing {s!r}; update it from results/"


def test_readme_verdict_is_current() -> None:
    p = ES["primary"]["pretrend"]["p"]
    assert p <= 0.10, "Test A now passes its pre-trend check; the verdict text needs revisiting"
    assert '**Overall verdict: "can\'t tell yet"**' in README


# Numbers and phrases from superseded runs. They may appear only in the Corrections section and the dated callout.
STALE = ["−4.1%, CI −8.1%", "Narrowly (pre-trend p = 0.107)", '**Overall verdict: "no effect larger', "21 high vs 43",
         "found 97% of the time", "+0.6% (CI −3.5%", "−6.8%, p = 0.087", "no effect larger than about 8% so far on"]


@pytest.mark.parametrize("s", STALE)
def test_no_stale_numbers_outside_corrections(s: str) -> None:
    body = README.split("## 13. Corrections")[0]
    body = body.split("*Results as of")[1] if "*Results as of" in body else body  # skip the dated revision callout
    assert s not in body


def test_scorecard_matches_results() -> None:
    a = ES["primary"]["delta_top_vs_bottom_quartile"]
    assert f"| {100 * a['est']:+.1f}% |" in SCORECARD
    assert f"pre-trend F-test p = {ES['primary']['pretrend']['p']:.3f} (**fails**" in SCORECARD
    assert "Verdict 4, can't tell yet" in SCORECARD
    assert "Verdict 3, no effect" not in SCORECARD
