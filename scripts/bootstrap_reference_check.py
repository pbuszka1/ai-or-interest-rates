"""Compare this project's wild cluster bootstrap with the `wildboottest` package (via pyfixest) on Test A.

The package tests one coefficient at a time, so this compares p-values for individual year coefficients.
Needs its own environment: pip install pyfixest wildboottest pyarrow. Output: results/bootstrap_reference_check.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import event_study as es  # noqa: E402
import event_study_industry as ei  # noqa: E402
import reestimate_pyfixest as rp  # noqa: E402

REPS = 9999


def main() -> None:
    q, x_names = rp.test_a_frame()
    q["cid"] = pd.factorize(q["id"])[0]                     # the bootstrap package needs numeric cluster ids
    fit = pf.feols(f"y ~ {' + '.join(x_names)} | id + t", q, vcov={"CRV1": "cid"})
    emp, base = ei.load()
    f, _, _, _ = ei.run(emp, base, ei.PRIMARY)
    out = {}
    for year in [2022, 2024, 2026, 2016]:
        name = f"X_{year}"
        ref = fit.wildboottest(param=name, reps=REPS, seed=7, impose_null=True, weights_type="rademacher")
        c = np.zeros(len(f.names))
        c[f.names.index(f"X:{year}")] = 1
        mine = es.wild_bootstrap_p(f, c, REPS, np.random.default_rng(7))
        out[name] = {"wildboottest_p": float(ref["Pr(>|t|)"]), "project_p": float(mine), "project_cluster_p": es.test(f, c)["p"],
                     "wildboottest_t": float(ref["t value"]), "coefficient_project": float(c @ f.beta), "coefficient_pyfixest": float(fit.coef()[name])}
        print(name, out[name], flush=True)
    (ROOT / "results" / "bootstrap_reference_check.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
