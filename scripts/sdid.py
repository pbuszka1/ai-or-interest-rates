"""Synthetic difference-in-differences, from scratch (Arkhangelsky, Athey, Hirshberg, Imbens
and Wager, "Synthetic Difference-in-Differences," American Economic Review, 2021).

Setup: a units x time matrix Y. Control units are never treated; treated units are treated
from the first post period on (block adoption).

The estimator combines two sets of weights:
  unit weights omega (over controls): make the weighted controls move in parallel with the
      treated average before treatment. An intercept is allowed, so they only need to match
      the *path*, not the level. A ridge penalty spreads weight across controls.
  time weights lambda (over pre periods): pick the pre-treatment periods that best predict
      the controls' post-treatment average, so the "before" baseline resembles "after".
Then a weighted difference-in-differences:
  tau = [treated post mean - lambda-weighted treated pre] - sum_j omega_j [control j post mean - lambda-weighted control j pre]

Inference: placebo method (their Algorithm 4). Repeatedly pretend a random set of controls,
the same size as the treated group, was treated; the spread of those fake estimates gives
the standard error. Needs more controls than treated units.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class SDIDResult:
    tau: float               # average effect over post periods
    path: np.ndarray         # effect at every period (pre periods should be near zero)
    omega: np.ndarray        # unit weights over controls
    lam: np.ndarray          # time weights over pre periods
    zeta: float              # ridge strength used for unit weights


def _simplex_ls(A: np.ndarray, b: np.ndarray, ridge: float) -> np.ndarray:
    """min ||(A - colmean) w + c - b||^2 + ridge ||w||^2 over w >= 0, sum w = 1, intercept c free.

    The intercept is handled by centering: the best c makes residuals average zero, which is
    the same as centering the columns of A and b over rows.
    """
    Ac, bc = A - A.mean(axis=0), b - b.mean()
    n = A.shape[1]
    res = minimize(lambda w: np.sum((Ac @ w - bc) ** 2) + ridge * np.sum(w ** 2), np.full(n, 1 / n),
                   jac=lambda w: 2 * Ac.T @ (Ac @ w - bc) + 2 * ridge * w,
                   bounds=[(0, 1)] * n, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   method="SLSQP", options={"maxiter": 2000, "ftol": 1e-14})
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def sdid(Y_co: np.ndarray, Y_tr: np.ndarray, T_pre: int, zeta: float | None = None) -> SDIDResult:
    """Y_co: controls x periods; Y_tr: treated x periods; the first T_pre periods are pre-treatment.

    zeta=None uses the paper's ridge strength; pass zeta=0 for no penalty (teaching only).
    """
    N_tr, T = Y_tr.shape
    T_post = T - T_pre
    pre, post = slice(0, T_pre), slice(T_pre, T)

    # Ridge strength from the paper: zeta = (N_tr * T_post)^(1/4) * sd of controls' pre-period changes.
    sigma = np.std(np.diff(Y_co[:, pre], axis=1), ddof=1)
    if zeta is None:
        zeta = (N_tr * T_post) ** 0.25 * sigma

    # Unit weights: rows = pre periods, columns = controls, target = treated average.
    omega = _simplex_ls(Y_co[:, pre].T, Y_tr[:, pre].mean(axis=0), ridge=zeta**2 * T_pre)
    # Time weights: rows = controls, columns = pre periods, target = each control's post mean.
    lam = _simplex_ls(Y_co[:, pre], Y_co[:, post].mean(axis=1), ridge=1e-6 * sigma**2 * Y_co.shape[0])

    tr = Y_tr.mean(axis=0)
    path = (tr - tr[pre] @ lam) - (omega @ Y_co - (omega @ Y_co[:, pre]) @ lam)
    return SDIDResult(float(path[post].mean()), path, omega, lam, float(zeta))


def summarize(r: SDIDResult, periods, T_pre: int) -> dict:
    """The diagnostics every SDID run reports, keyed for the results JSON:
    pre_path_rms        typical pre-period 'effect' (should be near 0: is the fit good?)
    effective_controls  1 / sum(omega^2): how many controls the weight is really spread over
    time_weights_over_1pct  which pre periods form the baseline (watch for one dominating)
    by_year, path       the effect by year (post period) and at every period."""
    post = pd.Series(r.path[T_pre:], index=periods[T_pre:])
    return {"pre_path_rms": round(float(np.sqrt(np.mean(r.path[:T_pre] ** 2))), 6),
            "effective_controls": round(float(1 / np.sum(r.omega ** 2)), 1),
            "time_weights_over_1pct": {f"{q:.2f}": round(float(v), 3) for q, v in zip(periods[:T_pre], r.lam) if v > 0.01},
            "by_year": {int(y): round(float(v), 6) for y, v in post.groupby(np.floor(post.index)).mean().items()},
            "path": {f"{q:.2f}": round(float(v), 6) for q, v in zip(periods, r.path)}}


def did(Y_co: np.ndarray, Y_tr: np.ndarray, T_pre: int) -> float:
    """Plain difference-in-differences with equal weights, for comparison."""
    pre, post = slice(0, T_pre), slice(T_pre, None)
    return float((Y_tr[:, post].mean() - Y_tr[:, pre].mean()) - (Y_co[:, post].mean() - Y_co[:, pre].mean()))


def jackknife_se(Y_co: np.ndarray, Y_tr: np.ndarray, T_pre: int,
                 cl_co: np.ndarray | None = None, cl_tr: np.ndarray | None = None) -> float:
    """Jackknife standard error (the paper's Algorithm 3), optionally by cluster.

    Weights are estimated once on the full sample. Each cluster is then left out in turn:
    remaining control weights are renormalised, time weights kept, and tau recomputed.
    Clusters let correlated units (one industry across regions) leave together, so the
    uncertainty isn't understated. Needs at least two treated clusters.
    """
    cl_co = np.arange(Y_co.shape[0]) if cl_co is None else np.asarray(cl_co)
    cl_tr = np.arange(Y_co.shape[0], Y_co.shape[0] + Y_tr.shape[0]) if cl_tr is None else np.asarray(cl_tr)
    full = sdid(Y_co, Y_tr, T_pre)
    pre, post = slice(0, T_pre), slice(T_pre, None)
    taus = []
    for c in np.unique(np.r_[cl_co, cl_tr]):
        keep_co, keep_tr = cl_co != c, cl_tr != c
        if keep_tr.sum() == 0 or full.omega[keep_co].sum() == 0:
            continue
        w = full.omega[keep_co] / full.omega[keep_co].sum()
        tr = Y_tr[keep_tr]
        d_tr = tr[:, post].mean() - tr[:, pre].mean(axis=0) @ full.lam
        d_co = w @ (Y_co[keep_co][:, post].mean(axis=1) - Y_co[keep_co][:, pre] @ full.lam)
        taus.append(d_tr - d_co)
    n = len(taus)
    return float(np.sqrt((n - 1) / n * np.sum((np.array(taus) - np.mean(taus)) ** 2)))


def placebo_se(Y_co: np.ndarray, N_tr: int, T_pre: int, reps: int = 200, seed: int = 0) -> float:
    """Placebo standard error: SDID on random fake-treated sets drawn from the controls."""
    rng = np.random.default_rng(seed)
    n = Y_co.shape[0]
    if n <= N_tr:
        raise ValueError("placebo inference needs more controls than treated units")
    taus = []
    for _ in range(reps):
        idx = rng.permutation(n)
        taus.append(sdid(Y_co[idx[N_tr:]], Y_co[idx[:N_tr]], T_pre).tau)
    return float(np.std(taus, ddof=1))
