"""Estimators must recover known answers before they touch real data.

Run from the project root: .venv/Scripts/python.exe -m pytest tests
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from sdid import did, jackknife_se, placebo_se, sdid  # noqa: E402
from synth import synth  # noqa: E402


def test_synthetic_control_recovers_lesson1_example():
    """Lesson 1: treated = 0.6 A + 0.4 B, effect -2 and -3 after period 4."""
    t = pd.Index([1.0, 2, 3, 4, 5, 6])
    donors = pd.DataFrame({"A": [10, 12, 14, 16, 18, 20], "B": [20, 21, 22, 23, 24, 25],
                           "C": [5, 5, 6, 6, 7, 7]}, index=t, dtype=float)
    treated = pd.Series([14.0, 15.6, 17.2, 18.8, 18.4, 19.0], index=t)
    r = synth(treated, donors, fit=(1, 4), demean=False)
    np.testing.assert_allclose(r.weights.to_numpy(), [0.6, 0.4, 0.0], atol=1e-4)
    np.testing.assert_allclose(r.gap.loc[[5.0, 6.0]].to_numpy(), [-2.0, -3.0], atol=1e-3)


def test_sdid_exact_without_penalty():
    """Lesson 2 hand example: with no ridge penalty the effect is exactly -3."""
    Y_co = np.array([[10, 11, 12, 13, 14, 15], [20, 22, 24, 26, 28, 30], [5, 5, 5, 5, 5, 5]], dtype=float)
    Y_tr = np.array([[50, 51.5, 53, 54.5, 53, 54.5]])
    assert sdid(Y_co, Y_tr, T_pre=4, zeta=0.0).tau == pytest.approx(-3.0, abs=1e-4)


def test_sdid_recovers_effect_under_factor_confounding():
    """Treated units load more on a time-varying factor, so parallel trends fails.
    SDID should still recover the true effect; plain DiD should not."""
    rng = np.random.default_rng(42)
    N_co, N_tr, T_pre, T_post, tau = 40, 8, 40, 12, -0.10
    T = T_pre + T_post
    factor = np.cumsum(rng.normal(0, 0.03, T))
    mu = np.r_[rng.normal(0, 1, N_co), rng.normal(1.5, 0.5, N_tr)]
    Y = rng.normal(0, 1, (N_co + N_tr, 1)) + np.cumsum(rng.normal(0.005, 0.01, T)) \
        + mu[:, None] * factor + rng.normal(0, 0.01, (N_co + N_tr, T))
    Y[N_co:, T_pre:] += tau
    est = sdid(Y[:N_co], Y[N_co:], T_pre)
    assert est.tau == pytest.approx(tau, abs=0.02)
    # No "effect" before treatment, on average. Not the maximum: SDID anchors its baseline on the
    # pre periods nearest treatment (time weights), so the earliest periods can drift a few points.
    assert np.sqrt(np.mean(est.path[:T_pre] ** 2)) < 0.03
    assert abs(est.path[T_pre - 4:T_pre]).max() < 0.02  # the year just before treatment is flat
    assert abs(did(Y[:N_co], Y[N_co:], T_pre) - tau) > abs(est.tau - tau)


def _event_panel(rng, delta=-0.04):
    """60 industries x quarters 2015-2026 (pandemic dropped), exposure gradient -delta per SD
    phased in from 2024, plus a rate trait with its own 2022 shock."""
    from event_study_industry import PANDEMIC
    q = np.arange(2015.0, 2026.75, 0.25)
    q = q[~((q >= PANDEMIC[0]) & (q <= PANDEMIC[1]))]
    ids = [f"i{j}" for j in range(60)]
    x, r = rng.normal(0, 1, 60), rng.normal(0, 1, 60)
    yr = np.floor(q)
    Y = rng.normal(0, 1, 60)[None, :] + np.cumsum(rng.normal(0, 0.01, len(q)))[:, None] \
        + np.outer(yr >= 2024, x) * delta + np.outer(yr >= 2022, r) * -0.03 + rng.normal(0, 0.01, (len(q), 60))
    return pd.DataFrame(Y, index=q, columns=ids), pd.Series(x, index=ids), pd.Series(r, index=ids)


def _dummy_ols(Y, traits, ref_year, block, w):
    """Brute force: weighted OLS with a dummy for every unit and every block x period."""
    T, N = Y.shape
    yr = np.floor(Y.index.to_numpy()).astype(int)
    years = [k for k in sorted(set(yr)) if k != ref_year]
    cols = [np.outer(yr == k, v.to_numpy()).ravel() for v in traits.values() for k in years]
    blk = pd.factorize(block)[0]
    period_blocks = [np.outer(np.arange(T) == s, blk == b).ravel().astype(float)
                     for b in np.unique(blk) for s in range(1, T)]       # period 0 omitted per block
    # Observations are period-major: row t*N + i is unit i in period t.
    D = np.column_stack(cols + [np.tile(np.eye(N), (T, 1))] + period_blocks)
    sw = np.sqrt(np.tile(w, T))
    return np.linalg.lstsq(D * sw[:, None], Y.to_numpy().ravel() * sw, rcond=None)[0][:len(cols)]


def test_event_study_demeaning_matches_dummy_regression():
    """Demeaning (Frisch-Waugh-Lovell) must equal OLS with all the dummies, both for plain
    unit + period effects (Test A) and for weighted unit + industry-by-period effects (Test B)."""
    import event_study as es
    Y, x, r = _event_panel(np.random.default_rng(1))
    f = es.fit(Y, {"X": x, "R": r}, 2019)
    one = pd.Series(0, index=Y.columns)
    np.testing.assert_allclose(f.beta, _dummy_ols(Y, {"X": x, "R": r}, 2019, one, np.ones(60)), atol=1e-8)

    rng = np.random.default_rng(2)
    block = pd.Series(rng.integers(0, 4, 60), index=Y.columns)
    w = pd.Series(rng.uniform(0.5, 3, 60), index=Y.columns)
    fb = es.fit(Y, {"X": x}, 2019, block=block, weight=w)
    np.testing.assert_allclose(fb.beta, _dummy_ols(Y, {"X": x}, 2019, block, w.to_numpy()), atol=1e-8)


def test_event_study_recovers_delta_and_separates_rate_shock():
    """Planted -4% per SD from 2024 must come back as Delta, and the 2022 rate shock on a
    different trait must not leak into it."""
    import event_study as es
    Y, x, r = _event_panel(np.random.default_rng(3))
    f = es.fit(Y, {"X": x, "R": r}, 2019)
    d = es.test(f, es.contrast(f, [2024, 2025, 2026], 2022))
    assert d["est"] == pytest.approx(-0.04, abs=0.006)
    assert d["ci95"][0] < -0.04 < d["ci95"][1]
    Yn, xn, rn = _event_panel(np.random.default_rng(4), delta=0.0)
    fn = es.fit(Yn, {"X": xn, "R": rn}, 2019)
    assert abs(es.test(fn, es.contrast(fn, [2024, 2025, 2026], 2022))["est"]) < 0.006


def test_prob_below_zero():
    """Flat prior: an estimate 1.96 SEs below zero is 97.5% likely below zero. A skeptical prior
    pulls that toward 50%, and more so the noisier the estimate."""
    import event_study as es
    assert es.prob_below_zero(-1.96, 1.0) == pytest.approx(0.975, abs=1e-3)
    assert 0.5 < es.prob_below_zero(-1.96, 1.0, prior_sd=1.0) < 0.975
    assert es.prob_below_zero(-1.96, 1.0, prior_sd=0.1) < es.prob_below_zero(-1.96, 1.0, prior_sd=1.0)


def test_jackknife_and_placebo_se_agree_roughly():
    """Two inference methods for the same estimate should give standard errors of the same
    order, and the jackknife interval should cover the true effect."""
    rng = np.random.default_rng(7)
    N_co, N_tr, T_pre, T_post, tau = 40, 10, 30, 8, -0.05
    T = T_pre + T_post
    Y = rng.normal(0, 1, (N_co + N_tr, 1)) + np.cumsum(rng.normal(0, 0.02, T)) + rng.normal(0, 0.03, (N_co + N_tr, T))
    Y[N_co:, T_pre:] += tau
    est = sdid(Y[:N_co], Y[N_co:], T_pre)
    jk = jackknife_se(Y[:N_co], Y[N_co:], T_pre)
    pl = placebo_se(Y[:N_co], N_tr, T_pre, reps=100)
    assert 0.33 < jk / pl < 3
    assert abs(est.tau - tau) < 2.5 * jk
