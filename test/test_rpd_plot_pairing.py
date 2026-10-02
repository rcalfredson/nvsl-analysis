"""Exercise plotRewards' RPD selection and tests without analyze.py's CLI startup."""

import ast
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from src.utils.common import propagate_nans, ttest_1samp, ttest_ind, ttest_rel


@pytest.fixture(scope="module")
def plot_function():
    source = ast.parse((Path(__file__).resolve().parents[1] / "analyze.py").read_text())
    return next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "plotRewards")


@pytest.mark.parametrize(
    "rpd,nf,roles,paired",
    [(True, 2, (0, 1), True), (True, 2, (0,), False),
     (True, 1, (0,), False), (False, 2, (0, 1), False)],
)
def test_rpd_plot_unions_bucket_exclusions_without_mutating_raw_values(
    plot_function, rpd, nf, roles, paired
):
    start = next(i for i, n in enumerate(plot_function.body)
                 if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "paired_rpd")
    selection = ast.Module(body=plot_function.body[start:start + 2], type_ignores=[])
    # One missing yoked bucket, one missing experimental bucket, valid zeros,
    # and another training whose complete pairs should remain available.
    raw = np.array([[[0., 10., np.nan, 2., np.nan, 3.],
                     [4., 5., 6., 7., 8., 9.]]])
    original = raw.copy()
    namespace = dict(a=raw, rpd=rpd, nf=nf, plot_fs=roles, propagate_nans=propagate_nans)
    exec(compile(selection, "analyze.py", "exec"), namespace)

    expected = original.copy()
    if paired:
        expected[0, 0, [1, 5]] = np.nan
    np.testing.assert_equal(namespace["a"], expected)
    np.testing.assert_equal(raw, original)
    assert namespace["paired_rpd"] is paired


@pytest.mark.parametrize("nosym", [True, False])
def test_rpd_bucket_test_compares_matched_flies_for_either_training_geometry(plot_function, nosym):
    expression = next(n.value for n in ast.walk(plot_function)
                      if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "tpm")
    exp = np.array([0., 10., 20., 100., np.nan])
    yok = np.array([0., 8., 15., np.nan, 200.])

    def get_values(group, bucket=None, delta=False, f1=None):
        if delta:
            return exp - yok
        return exp if f1 == 0 else yok

    namespace = dict(np=np, nb=1, cmpg=False, dlt=nosym, paired_rpd=True,
                     getVals=get_values, ttest_rel=ttest_rel,
                     ttest_ind=ttest_ind, ttest_1samp=ttest_1samp)
    result = eval(compile(ast.Expression(expression), "analyze.py", "eval"), namespace)
    expected = stats.ttest_rel(exp[:3], yok[:3])
    np.testing.assert_allclose(result[:2, 0], [expected.statistic, expected.pvalue])


def test_rpd_between_group_bucket_test_remains_unpaired(plot_function):
    expression = next(n.value for n in ast.walk(plot_function)
                      if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "tpm")
    groups = [np.array([0., 3., 7.]), np.array([10., 14., 19., 25.])]
    namespace = dict(np=np, nb=1, cmpg=True, dlt=True, paired_rpd=True,
                     getVals=lambda g, *args: groups[g], ttest_ind=ttest_ind,
                     ttest_rel=ttest_rel, ttest_1samp=ttest_1samp)
    result = eval(compile(ast.Expression(expression), "analyze.py", "eval"), namespace)
    np.testing.assert_allclose(result[:2, 0], ttest_ind(*groups)[:2])
