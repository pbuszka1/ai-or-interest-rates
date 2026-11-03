"""Estimate each occupation group's sensitivity to interest-rate shocks and to the cycle, 1990-2019.

Both are measured *before* generative AI existed, so AI can't contaminate them.

Rate sensitivity. For group g, relative growth y(g,t) = dlog E(g,t) - dlog E(all,t), i.e. how
much faster or slower the group grew than total employment that quarter, is regressed on the
current and 7 previous quarters of monetary policy surprises (Bauer-Swanson MPS_ORTH, summed
to quarters; positive = tighter than expected). The sum of the 8 coefficients is the group's
cumulative relative response over two years to a 1-point surprise. Standard errors are
Newey-West (4 lags), because quarterly growth rates are autocorrelated.

Cyclicality. dlog E(g,t) regressed on dlog E(all,t): 1 means the group moves one-for-one
with total employment, above 1 means it swings harder.

Quarters where the CPS changed its design or occupation codes are dropped, because the
growth across them is measurement, not economics: 1994-Q1 (survey redesign), 2003-Q1
(new occupation codes), 2011-Q1 (2010 Census codes).

Group-level CPS growth is noisy, so raw estimates are also shrunk toward the average
(empirical Bayes): noisier estimates move further toward the mean.

Output: data/processed/occ_rate_sensitivity.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
SHOCKS = ROOT / "data" / "raw" / "shocks" / "frbsf_monetary_policy_surprises_2026-10-22.xlsx"
LAGS = 8
BREAKS = {1994.0, 2003.0, 2011.0}  # t values of 1994-Q1, 2003-Q1, 2011-Q1
MIN_EMPLOYED = 50_000              # skip groups too small for a usable growth series


def quarterly_shocks(kind: str = "mps") -> pd.Series:
    """'mps': Bauer-Swanson surprises (clean, small). 'gs2': quarterly change in the 2-year
    Treasury yield (moves far more, but partly reacts to the economy, so it's a power check,
    not a causal measure)."""
    if kind == "gs2":
        g = pd.read_csv(ROOT / "data" / "raw" / "fred" / "GS2.csv", parse_dates=["date"])
        g["t"] = g["date"].dt.year + ((g["date"].dt.month - 1) // 3) / 4
        return g.groupby("t")["value"].mean().diff()
    m = pd.read_excel(SHOCKS, sheet_name="Monthly (update 2023)")
    m["t"] = m["Year"] + ((m["Month"] - 1) // 3) / 4
    return m.groupby("t")["MPS_ORTH"].sum()


def shrink(est: pd.Series, se: pd.Series) -> pd.Series:
    """Empirical Bayes shrinkage toward the precision-weighted mean.

    tau^2 is the spread of true values across groups: the variance of the estimates minus the
    average sampling variance (floored at zero).
    """
    ok = est.notna() & se.notna()
    mean = np.average(est[ok], weights=1 / se[ok] ** 2)
    tau2 = max(est[ok].var() - (se[ok] ** 2).mean(), 0.0)
    return mean + tau2 / (tau2 + se**2) * (est - mean)  # NaN stays NaN for units without an estimate


def main(kind: str = "mps") -> None:
    p = pd.read_parquet(PROC / "panel_occ_age_quarter.parquet")
    groups = pd.read_csv(PROC / "occ_groups.csv")
    keep = groups.loc[groups["employed_2015_19"] >= MIN_EMPLOYED, "group"]

    emp = p[(p["t"] >= 1990) & (p["t"] < 2020)].groupby(["t", "group"])["employed"].sum().unstack()
    dlog = np.log(emp).diff()
    dlog_all = np.log(emp.sum(axis=1)).diff()
    shocks = quarterly_shocks(kind).reindex(dlog.index)
    X = pd.concat({f"s{k}": shocks.shift(k) for k in range(LAGS)}, axis=1)
    valid = ~dlog.index.isin(BREAKS) & X.notna().all(axis=1) & dlog_all.notna()

    rows = []
    for g in keep:
        if g not in dlog:
            continue
        y = (dlog[g] - dlog_all)[valid]
        ok = y.notna()
        if ok.sum() < 60:
            continue
        fit = sm.OLS(y[ok], sm.add_constant(X[valid][ok])).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
        lag_cols = [f"s{k}" for k in range(LAGS)]
        cum = fit.params[lag_cols].sum()
        cum_se = np.sqrt(np.ones(LAGS) @ fit.cov_params().loc[lag_cols, lag_cols].to_numpy() @ np.ones(LAGS))
        cyc = sm.OLS(dlog[g][valid][ok], sm.add_constant(dlog_all[valid][ok])).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
        rows.append({"group": g, "rate_sensitivity_raw": cum, "rate_sensitivity_se": cum_se,
                     "cyclicality_raw": cyc.params.iloc[1], "cyclicality_se": cyc.bse.iloc[1],
                     "n_quarters": int(ok.sum())})

    r = pd.DataFrame(rows).set_index("group")
    r["rate_sensitivity"] = shrink(r["rate_sensitivity_raw"], r["rate_sensitivity_se"])
    r["cyclicality"] = shrink(r["cyclicality_raw"], r["cyclicality_se"])
    suffix = "" if kind == "mps" else f"_{kind}"
    r.reset_index().to_csv(PROC / f"occ_rate_sensitivity{suffix}.csv", index=False)

    print(f"{len(r)} groups estimated over 1990-2019 ({int(r['n_quarters'].median())} quarters each, median)")
    sig = (r["rate_sensitivity_raw"].abs() > 1.96 * r["rate_sensitivity_se"]).mean()
    print(f"share of groups with rate sensitivity significant at 5%: {sig:.0%} (5% expected by chance)")
    print(f"spread of raw estimates (sd) {r['rate_sensitivity_raw'].std():.3f} vs median s.e. {r['rate_sensitivity_se'].median():.3f}")
    print(f"correlation of raw rate sensitivity with cyclicality: {r['rate_sensitivity_raw'].corr(r['cyclicality_raw']):.2f}")
    m = r.join(groups.set_index("group")[["dv_rating_beta", "human_rating_beta"]])
    print(f"correlation with AI exposure (GPT-4 beta): rate {m['rate_sensitivity'].corr(m['dv_rating_beta']):.2f}, "
          f"cyclicality {m['cyclicality'].corr(m['dv_rating_beta']):.2f}")
    show = ["15-12", "15-20", "13-20", "43-40", "47-20", "41-90", "29-11", "35-20"]
    print("\nselected groups:")
    print(r.loc[r.index.intersection(show), ["rate_sensitivity_raw", "rate_sensitivity_se", "rate_sensitivity",
                                              "cyclicality_raw", "cyclicality"]].round(3).to_string())


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
