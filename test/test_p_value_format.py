from src.plotting.p_value_format import (
    format_plot_p_value,
    use_mathtext_exponents,
)


def test_format_plot_p_value_replaces_negative_e_exponent_with_mathtext():
    assert format_plot_p_value(1.234e-5) == r"$1.23 \times 10^{-5}$"


def test_format_plot_p_value_leaves_decimal_notation_unchanged():
    assert format_plot_p_value(0.00321) == "0.00321"


def test_use_mathtext_exponents_handles_positive_and_zero_padded_exponents():
    assert use_mathtext_exponents("1E+06 and 2e-003") == (
        r"$1 \times 10^{6}$ and $2 \times 10^{-3}$"
    )
