"""Synthetic control, from scratch: Abadie, Diamond and Hainmueller (2010).

Weights are non-negative and sum to one, chosen to minimise squared pre-period error.
`demean=True` subtracts each series' own fit-window mean first (the "demeaned" synthetic
control of Ferman and Pinto), so the blend has to match the treated unit's *path*, not its
level. That matters here: no blend of smaller occupations can match the level of the largest
group, and the level isn't what we care about.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class SynthResult:
    weights: pd.Series      # donor -> weight
    synthetic: pd.Series    # time -> synthetic outcome (same scale as the treated series)
    gap: pd.Series          # time -> treated minus synthetic
    rmspe_fit: float        # typical gap inside the fit window
    rmspe_post: float       # typical gap after the fit window


def fit_weights(y: np.ndarray, X: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    """min ||y - X w||^2 + ridge * ||w||^2 subject to w >= 0, sum(w) = 1.

    The ridge term penalises concentrating weight on a few donors, spreading it more evenly
    (the regularisation used in synthetic DiD and penalised synthetic control).
    """
    J = X.shape[1]
    res = minimize(lambda w: np.sum((y - X @ w) ** 2) + ridge * np.sum(w ** 2), np.full(J, 1 / J),
                   jac=lambda w: -2 * X.T @ (y - X @ w) + 2 * ridge * w,
                   bounds=[(0, 1)] * J, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   method="SLSQP", options={"maxiter": 1000, "ftol": 1e-12})
    return np.clip(res.x, 0, None) / np.clip(res.x, 0, None).sum()


def synth(treated: pd.Series, donors: pd.DataFrame, fit: tuple[float, float],
          post_end: float | None = None, demean: bool = True,
          exclude: tuple[float, float] | None = None, ridge: float = 0.0) -> SynthResult:
    """Fit on times in [fit[0], fit[1]], minus any `exclude` window (e.g. the pandemic);
    report the gap for every time in the data up to post_end."""
    idx = treated.index if post_end is None else treated.index[treated.index <= post_end]
    y, X = treated.loc[idx], donors.loc[idx]
    in_fit = (idx >= fit[0]) & (idx <= fit[1])
    if exclude is not None:
        in_fit &= ~((idx >= exclude[0]) & (idx <= exclude[1]))
    y_c, X_c = y.copy(), X.copy()
    if demean:
        y_c = y - y[in_fit].mean()
        X_c = X - X[in_fit].mean()
    w = fit_weights(y_c[in_fit].to_numpy(), X_c[in_fit].to_numpy(), ridge)
    synth_c = X_c.to_numpy() @ w
    synthetic = pd.Series(synth_c + (y[in_fit].mean() if demean else 0.0), index=idx)
    gap = y - synthetic
    after = idx > fit[1]
    return SynthResult(pd.Series(w, index=donors.columns), synthetic, gap,
                       float(np.sqrt(np.mean(gap[in_fit] ** 2))),
                       float(np.sqrt(np.mean(gap[after] ** 2))) if after.any() else float("nan"))


def placebos(series: pd.DataFrame, donors: list[str], fit: tuple[float, float],
             post_end: float | None = None, exclude: tuple[float, float] | None = None,
             ridge: float = 0.0) -> pd.DataFrame:
    """Rerun synthetic control with each donor as the 'treated' unit and the other donors as its pool."""
    rows = {}
    for d in donors:
        others = [o for o in donors if o != d]
        r = synth(series[d], series[others], fit, post_end, exclude=exclude, ridge=ridge)
        rows[d] = {"rmspe_fit": r.rmspe_fit, "rmspe_post": r.rmspe_post,
                   "ratio": r.rmspe_post / r.rmspe_fit, "gap": r.gap}
    return pd.DataFrame(rows).T


def p_value(ratio: float, placebo: pd.DataFrame) -> float:
    """Permutation p-value: share of all units (treated plus placebos) with a post/pre RMSPE
    ratio at least as large as the treated unit's."""
    return float((1 + (placebo["ratio"] >= ratio).sum()) / (1 + len(placebo)))


def plot(treated: pd.Series, result: SynthResult, placebo: pd.DataFrame, fit: tuple[float, float],
         events: list[tuple[float, str]], title: str, label: str, path_stem: str,
         start: float | None = None) -> None:
    """Two panels: treated vs synthetic (top), gap with placebo gaps in grey (bottom).
    Saves <path_stem>.svg and .png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    syn, gap = result.synthetic, result.gap
    if start is not None:
        syn, gap = syn[syn.index >= start], gap[gap.index >= start]
    t = syn.index
    base = treated.loc[t[(t >= fit[0]) & (t <= fit[1])]].mean()
    fig, (a, b) = plt.subplots(2, 1, figsize=(8, 7), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    a.plot(t, 100 * (treated.loc[t] - base), color="#1f4e79", lw=1.8, label=label)
    a.plot(t, 100 * (syn - base), color="#c0504d", lw=1.8, ls="--", label="Synthetic (low-exposure blend)")
    a.set_ylabel("Log employment vs fit-period mean (×100)")
    a.legend(frameon=False, loc="upper left")
    for g in placebo["gap"]:
        g = g if start is None else g[g.index >= start]
        b.plot(g.index, 100 * g, color="0.8", lw=0.7)
    b.plot(t, 100 * gap, color="#1f4e79", lw=1.8)
    b.axhline(0, color="0.3", lw=0.8)
    b.set_ylabel("Gap (×100 log points)")
    for ax in (a, b):
        ax.axvspan(fit[1] + 0.25, t.max() + 0.25, color="0.95", zorder=0)
        for x, _ in events:
            ax.axvline(x, color="0.4", lw=0.8, ls=":")
    lo, hi = a.get_ylim()
    for i, (x, lab) in enumerate(events):
        a.text(x, lo + (0.12 + 0.08 * i) * (hi - lo), f" {lab}", va="bottom", fontsize=8, color="0.3")
    b.set_xlabel("Year (shaded = after the fit window; grey lines = placebo units)")
    a.set_title(title, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{path_stem}.svg")
    fig.savefig(f"{path_stem}.png", dpi=120)
    plt.close(fig)
