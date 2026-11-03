"""Shared estimator for the Round 2 event studies (Tests A and B in docs/PREREGISTRATION.md).

Model, for unit i (an industry, or an occupation-in-industry cell) in period t:

    log Y(i,t) = a(i) + d(block(i), t) + sum over years k of  theta_k * X(i) * 1[year(t) = k]
                                       + (the same year-by-year terms for each control trait)

    a(i)            unit fixed effect: each unit's normal level
    d(block, t)     period effect, shared by all units in the same block. Test A has one block
                    (plain quarter effects); Test B uses industries as blocks, so anything that
                    hit an industry in a year cancels out
    theta_k         the result: how much a unit one unit of X higher sits in year k, relative to
                    the reference year

How it's computed: the panel must be balanced (every unit observed in every period). Then the
fixed effects are removed exactly by demeaning (Frisch-Waugh-Lovell), and ordinary (weighted)
least squares on what's left gives theta. Standard errors are clustered, so a cluster's
periods (and units) count as one piece of evidence, not many.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class Fit:
    beta: np.ndarray        # coefficients, one per name
    vcov: np.ndarray        # clustered covariance matrix of beta
    names: list[str]        # "trait:year", e.g. "X:2024"
    X: np.ndarray           # demeaned, weighted regressors (rows = period-major observations)
    y: np.ndarray           # demeaned, weighted outcome
    A: np.ndarray           # (X'X)^-1, reused by the bootstrap
    codes: np.ndarray       # cluster id of each observation
    G: int                  # number of clusters
    k_fe: int               # number of fixed effects absorbed (for the small-sample correction)


def demean(Z: np.ndarray, block: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Remove unit means and (weighted) block-by-period means from a periods x units matrix.

    For a balanced panel with weights fixed per unit, this is exactly what regressing on unit
    and block-by-period dummies would remove."""
    out = np.empty_like(Z)
    for b in np.unique(block):
        m = block == b
        Zb, wb = Z[:, m], w[m]
        unit_mean = Zb.mean(0)                                  # each unit's average over time
        period_mean = Zb @ wb / wb.sum()                        # block's weighted average each period
        grand = unit_mean @ wb / wb.sum()
        out[:, m] = Zb - unit_mean[None, :] - (period_mean - grand)[:, None]
    return out


def _vcov(X: np.ndarray, u: np.ndarray, A: np.ndarray, codes: np.ndarray, G: int, k_fe: int) -> np.ndarray:
    """Cluster-robust (CR1) covariance: sum each cluster's score, then sandwich."""
    S = np.zeros((G, X.shape[1]))
    np.add.at(S, codes, X * u[:, None])
    n, k = X.shape
    c = G / (G - 1) * (n - 1) / (n - k - k_fe)                  # the usual small-sample correction
    return c * A @ (S.T @ S) @ A


def fit(Y: pd.DataFrame, traits: dict[str, pd.Series], ref_year: int,
        block: pd.Series | None = None, cluster: pd.Series | None = None,
        weight: pd.Series | None = None) -> Fit:
    """Y: periods x units (index = decimal time, e.g. 2024.25). traits: name -> value per unit.
    block / cluster / weight: per-unit labels (default: one block, cluster by unit, equal weight)."""
    T, N = Y.shape
    units = Y.columns
    for label, s in [("block", block), ("cluster", cluster), ("weight", weight)]:
        if s is not None and s.reindex(units).isna().any():
            raise ValueError(f"{label} has missing values for some units")  # would silently merge them
    blk = np.zeros(N) if block is None else pd.factorize(block.reindex(units))[0]
    cl = np.arange(N) if cluster is None else pd.factorize(cluster.reindex(units))[0]
    w = np.ones(N) if weight is None else weight.reindex(units).to_numpy(float)
    sw = np.tile(np.sqrt(w), T)                                 # weights enter OLS as sqrt(w) on each row

    year = np.floor(Y.index.to_numpy()).astype(int)
    years = [k for k in sorted(set(year)) if k != ref_year]     # the reference year is the omitted one
    cols, names = [], []
    for name, v in traits.items():
        vals = v.reindex(units).to_numpy(float)
        for k in years:
            cols.append(demean(np.outer(year == k, vals), blk, w).ravel() * sw)
            names.append(f"{name}:{k}")
    X = np.column_stack(cols)
    y = demean(Y.to_numpy(float), blk, w).ravel() * sw

    A = np.linalg.inv(X.T @ X)
    beta = A @ X.T @ y
    codes, G = np.tile(cl, T), int(cl.max()) + 1
    k_fe = N + len(np.unique(blk)) * (T - 1)
    return Fit(beta, _vcov(X, y - X @ beta, A, codes, G, k_fe), names, X, y, A, codes, G, k_fe)


def contrast(f: Fit, post_years: list[int], base_year: int, var: str = "X") -> np.ndarray:
    """Weights that turn the coefficients into Delta = mean(theta over post_years) - theta[base_year]."""
    c = np.zeros(len(f.names))
    for k in post_years:
        c[f.names.index(f"{var}:{k}")] += 1 / len(post_years)
    c[f.names.index(f"{var}:{base_year}")] -= 1
    return c


def test(f: Fit, c: np.ndarray, scale: float = 1.0) -> dict:
    """Estimate, standard error, 95% CI and two-sided p for the combination c'beta, times `scale`.
    Uses a t distribution with (clusters - 1) degrees of freedom."""
    est, se = scale * float(c @ f.beta), abs(scale) * float(np.sqrt(c @ f.vcov @ c))
    tq = stats.t.ppf(0.975, f.G - 1)
    return {"est": est, "se": se, "ci95": [est - tq * se, est + tq * se],
            "p": float(2 * stats.t.sf(abs(est / se), f.G - 1))}


def prob_below_zero(est: float, se: float, prior_sd: float | None = None) -> float:
    """Probability the true effect is below zero, given the estimate and its standard error.

    With no prior (flat), the belief about the true effect is Normal(est, se), so the answer is
    Phi(-est/se). With a skeptical prior Normal(0, prior_sd), the estimate is first shrunk toward
    zero: posterior mean = est * prior_sd^2 / (prior_sd^2 + se^2). Either way it assumes the model
    is right; it says nothing about confounding."""
    if prior_sd is None:
        return float(stats.norm.cdf(-est / se))
    shrink = prior_sd ** 2 / (prior_sd ** 2 + se ** 2)
    return float(stats.norm.cdf(-est * shrink / np.sqrt(shrink * se ** 2)))


def wild_bootstrap_p(f: Fit, c: np.ndarray, reps: int, rng) -> float:
    """Wild cluster restricted bootstrap: impose c'beta = 0, rebuild the outcome with each
    cluster's residuals randomly sign-flipped, and count bootstrap |t| >= the real |t|."""
    b_r = f.beta - f.A @ c * (c @ f.beta) / (c @ f.A @ c)       # closest coefficients with c'beta = 0
    u_r = f.y - f.X @ b_r
    t0 = (c @ f.beta) / np.sqrt(c @ f.vcov @ c)
    hits = 0
    for _ in range(reps):
        ys = f.X @ b_r + u_r * rng.choice([-1.0, 1.0], f.G)[f.codes]
        bs = f.A @ f.X.T @ ys
        V = _vcov(f.X, ys - f.X @ bs, f.A, f.codes, f.G, f.k_fe)
        hits += abs((c @ bs) / np.sqrt(c @ V @ c)) >= abs(t0)
    return hits / reps


def pretrend(f: Fit, years: list[int], var: str = "X") -> dict:
    """Joint Wald F-test that theta is zero in every pre-period year."""
    idx = [f.names.index(f"{var}:{k}") for k in years if f"{var}:{k}" in f.names]
    b, V = f.beta[idx], f.vcov[np.ix_(idx, idx)]
    F = float(b @ np.linalg.solve(V, b) / len(idx))
    return {"years": years, "F": F, "p": float(stats.f.sf(F, len(idx), f.G - 1))}


def path(f: Fit, ref_year: int, var: str = "X") -> dict[int, dict]:
    """theta and its standard error for every year, with the reference year at zero."""
    out = {ref_year: {"theta": 0.0, "se": 0.0}}
    for i, n in enumerate(f.names):
        v, k = n.split(":")
        if v == var:
            out[int(k)] = {"theta": float(f.beta[i]), "se": float(np.sqrt(f.vcov[i, i]))}
    return dict(sorted(out.items()))


def trend_adjusted(f: Fit, pre_years: list[int], ref_year: int, post_years: list[int], base_year: int) -> np.ndarray:
    """Contrast for Delta after removing a straight-line pre-trend (Rambachan-Roth with M = 0):
    fit a line through the pre-period thetas (anchored at 0 in the reference year), extrapolate
    it over the gap between base_year and the post years, and subtract."""
    d = np.array([k - ref_year for k in pre_years], float)
    slope = np.zeros(len(f.names))
    for k, dk in zip(pre_years, d):
        slope[f.names.index(f"X:{k}")] = dk / np.sum(d ** 2)   # least-squares slope through the origin
    return contrast(f, post_years, base_year) - (np.mean(post_years) - base_year) * slope


def quartile_gap(x: pd.Series) -> float:
    """Average exposure of the top quarter minus the bottom quarter: turns 'per SD' into
    'most vs least exposed'."""
    return float(x[x >= x.quantile(0.75)].mean() - x[x <= x.quantile(0.25)].mean())


def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=1)
