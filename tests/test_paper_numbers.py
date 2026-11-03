"""Every number in the paper must match the saved results to the precision the paper prints."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import verify_paper_numbers as vp  # noqa: E402

ROWS = vp.run()


@pytest.mark.parametrize("label,shown,paper,ok", ROWS, ids=[r[0] for r in ROWS])
def test_number(label: str, shown: float, paper: float, ok: bool) -> None:
    assert ok, f"{label}: results give {shown}, the paper prints {paper}"
