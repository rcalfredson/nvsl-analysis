import numpy as np
import pytest

from src.analysis.repeated_measures_anova import repeated_measures_anova_tukey


def test_repeated_measures_and_tukey_match_unbalanced_r_reference():
    groups = {
        "A": [[1, 2, 4], [3, 4, 5], [2, 4, 4]],
        "B": [[2, 2, 3], [3, 4, 4], [4, 4, 5], [4, 5, 5]],
        "C": [[4, 5, 7], [5, 6, 7], [6, 7, 9], [4, 6, 6], [7, 7, 8]],
    }
    effects, comparisons = repeated_measures_anova_tukey(groups)

    # R lm on subject means and orthonormal within-subject contrasts.
    assert [(e.df_numerator, e.df_denominator) for e in effects] == [(2, 9), (2, 18), (4, 18)]
    np.testing.assert_allclose(
        [e.statistic for e in effects],
        [11.6090986575751, 40.8069752138627, 2.2847938144330],
        rtol=1e-12,
    )

    # R lm cell residuals with ptukey and qtukey.
    contrast = next(result for result in comparisons[2] if result.group_b == "C")
    assert (contrast.group_a, contrast.n_a, contrast.group_b, contrast.n_b) == ("A", 3, "C", 5)
    assert contrast.df == 27
    assert contrast.statistic == pytest.approx(5.598173826525, rel=1e-10)
    assert contrast.p_value_adjusted == pytest.approx(0.00138990156, abs=1e-9)
    assert (contrast.ci95_low, contrast.ci95_high) == pytest.approx(
        (-4.98747878, -1.14585455), abs=1e-7
    )


def test_repeated_measures_requires_complete_subject_rows():
    with pytest.raises(ValueError, match="complete observations"):
        repeated_measures_anova_tukey({"A": [[1, 2], [2, np.nan]], "B": [[3, 4], [4, 5]]})
