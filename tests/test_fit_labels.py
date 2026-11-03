"""fit() must refuse missing block labels, which would otherwise merge every unit into one block."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import event_study as es  # noqa: E402


def panel() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    units = [f"u{i}" for i in range(8)]
    y = pd.DataFrame(rng.normal(size=(4, 8)), index=[2018.0, 2019.0, 2020.0, 2021.0], columns=units)
    return y, pd.Series(np.arange(8.0), index=units)


def test_missing_block_labels_are_refused() -> None:
    y, x = panel()
    block = pd.Series([np.nan] * 8, index=y.columns)
    with pytest.raises(ValueError, match="block"):
        es.fit(y, {"X": x}, 2019, block=block)


def test_block_labels_change_the_fit() -> None:
    y, x = panel()
    block = pd.Series(["a"] * 4 + ["b"] * 4, index=y.columns)
    assert not np.allclose(es.fit(y, {"X": x}, 2019).beta, es.fit(y, {"X": x}, 2019, block=block).beta)
