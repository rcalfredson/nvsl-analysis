from types import SimpleNamespace

import numpy as np
import pytest

from src.utils.debug_fly_groups import write_sorted_fly_list


def test_write_sorted_fly_list_uses_mask_and_sorts(tmp_path):
    vas = [
        SimpleNamespace(fn="/data/zeta.avi", f=2),
        SimpleNamespace(fn="/data/alpha.avi", f=1),
        SimpleNamespace(fn="/data/beta.avi", f=None),
    ]
    out = tmp_path / "cohort.txt"

    write_sorted_fly_list(out, np.array([True, True, False]), vas)

    assert out.read_text(encoding="utf-8") == (
        "alpha.avi\tfly=1\n"
        "zeta.avi\tfly=2\n"
    )


def test_write_sorted_fly_list_rejects_misaligned_mask(tmp_path):
    vas = [SimpleNamespace(fn="a.avi", f=1)]

    with pytest.raises(ValueError, match="one entry per VA"):
        write_sorted_fly_list(tmp_path / "cohort.txt", [True, False], vas)
