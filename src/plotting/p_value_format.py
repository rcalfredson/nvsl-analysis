from __future__ import annotations

import re


_E_NOTATION_RE = re.compile(
    r"(?P<mantissa>[+-]?(?:\d+(?:\.\d*)?|\.\d+))[eE](?P<exponent>[+-]?\d+)"
)


def use_mathtext_exponents(text: str) -> str:
    """Replace E-notation numbers with MathText powers of ten."""

    def replace(match: re.Match[str]) -> str:
        exponent = int(match.group("exponent"))
        return rf'${match.group("mantissa")} \times 10^{{{exponent}}}$'

    return _E_NOTATION_RE.sub(replace, str(text))


def format_plot_p_value(p_value: float, format_spec: str = ".3g") -> str:
    """Format a numeric p-value without E-notation for use in plot text."""

    return use_mathtext_exponents(format(float(p_value), format_spec))
