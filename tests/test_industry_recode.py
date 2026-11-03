"""The 2012-to-2017 industry recode must keep total weight and reach only 2017 codes."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_industry_panel import recode_2012_industries  # noqa: E402


def test_recode_keeps_weight_and_handles_merges_and_splits() -> None:
    cps = pd.DataFrame({"IND": [770, 1680, 1690, 8560], "WTFINL": [10.0, 3.0, 1.0, 8.0]})
    out = recode_2012_industries(cps)
    assert out["WTFINL"].sum() == cps["WTFINL"].sum()
    weights = out.groupby("IND")["WTFINL"].sum()
    assert weights[770] == 10.0                       # unchanged code stays put
    assert weights[1691] == 4.0                       # two codes merge
    assert all(weights[k] == 2.0 for k in [8561, 8562, 8563, 8564])  # one code splits evenly
