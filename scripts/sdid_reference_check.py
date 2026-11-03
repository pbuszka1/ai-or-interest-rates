"""Compare this project's SDiD with the `synthdid` Python package (a port of the reference R package).

Needs its own environment: pip install synthdid scipy. Reads data/processed/sdid_reference_input_*.csv, written by
scripts/verification_checks.py. Time is passed as an integer period index: the package misreads fractional quarters.
Output: results/sdid_reference_check.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from synthdid.synthdid import Synthdid

ROOT = Path(__file__).resolve().parents[1]
RES, PROC = ROOT / "results", ROOT / "data" / "processed"
sys.path.insert(0, str(ROOT / "scripts"))
from sdid import sdid  # noqa: E402


def main() -> None:
    saved = json.loads((RES / "sdid_industry.json").read_text(encoding="utf-8"))
    out = {}
    for name in ["2022_high_pre_ends_2019", "2022_high"]:
        d = pd.read_csv(PROC / f"sdid_reference_input_{name}.csv")
        times = sorted(d["time"].unique())
        d["t"] = d["time"].map({v: i for i, v in enumerate(times)})
        ref = Synthdid(d.drop(columns="time"), unit="unit", time="t", treatment="treatment", outcome="outcome").fit()
        T_pre = int(ref.att_info["T0"].iloc[0])
        treated = sorted(d.loc[d.treatment == 1, "unit"].unique())
        wide = d.pivot(index="unit", columns="t", values="outcome")
        controls = [u for u in wide.index if u not in treated]
        mine = sdid(wide.loc[controls].to_numpy(), wide.loc[treated].to_numpy(), T_pre)
        r_om = pd.Series(np.asarray(ref.weights["omega"][0]).ravel(), index=sorted(controls))
        m_om = pd.Series(mine.omega, index=controls).reindex(r_om.index)
        r_lam, m_lam = np.asarray(ref.weights["lambda"][0]).ravel(), mine.lam
        out[name] = {"T_pre": T_pre, "reference_tau": float(ref.att_info["att_time"].iloc[0]), "project_tau_recomputed": mine.tau,
                     "project_tau_saved": saved[name]["tau"],
                     "unit_weight_correlation": float(np.corrcoef(r_om, m_om)[0, 1]), "unit_weight_max_abs_difference": float((r_om - m_om).abs().max()),
                     "time_weight_max_abs_difference": float(np.abs(r_lam - m_lam).max()),
                     "time_weights_reference": {f"{times[i]:.2f}": round(float(r_lam[i]), 3) for i in np.nonzero(r_lam > 0.01)[0]},
                     "time_weights_project": {f"{times[i]:.2f}": round(float(m_lam[i]), 3) for i in np.nonzero(m_lam > 0.01)[0]},
                     "weights_valid": bool(abs(r_om.sum() - 1) < 1e-8 and abs(m_om.sum() - 1) < 1e-8 and (m_om >= 0).all() and (m_lam >= 0).all() and abs(m_lam.sum() - 1) < 1e-8)}
        print(name, json.dumps(out[name], indent=1))
    (RES / "sdid_reference_check.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
