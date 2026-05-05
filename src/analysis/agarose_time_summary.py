from __future__ import annotations

import csv
import html
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_SECTION = (
    "% time over agarose (contact events begin when edge of fitted ellipse crosses "
    "agarose border)"
)
DEFAULT_PRE_COL = "% time spent on agarose- pre last 10m (exp)"
DEFAULT_POST_COL = "% time spent on agarose- T3 post last 10m (exp)"
DEFAULT_SUMMARY_COLS = (
    DEFAULT_PRE_COL,
    "% time spent on agarose- T1 start (exp)",
    "% time spent on agarose- T3 end (exp)",
    DEFAULT_POST_COL,
)

MISSING_STRINGS = {"", "nan", "na", "n/a", "none", "null"}


@dataclass(frozen=True)
class DescriptiveStats:
    n: int
    mean: float
    sd: float
    sem: float
    ci95_low: float
    ci95_high: float


@dataclass(frozen=True)
class PairedTest:
    group: str
    pre_col: str
    post_col: str
    n_pairs: int
    mean_pre: float
    mean_post: float
    mean_difference: float
    ci95_low: float
    ci95_high: float
    t_stat: float
    df: float
    p_value: float
    reductions: np.ndarray


@dataclass(frozen=True)
class WelchTest:
    group_a: str
    group_b: str
    n_a: int
    n_b: int
    mean_reduction_a: float
    mean_reduction_b: float
    mean_difference_a_minus_b: float
    ci95_low: float
    ci95_high: float
    t_stat: float
    df: float
    p_value: float
    test: str


@dataclass(frozen=True)
class ParsedGroup:
    label: str
    path: str
    data: pd.DataFrame
    coercion_warnings: dict[str, int]


def available_section_titles(path: str | Path) -> list[str]:
    lines = Path(path).read_text().splitlines()
    titles: list[str] = []
    for i, line in enumerate(lines[:-1]):
        stripped = line.strip().strip('"')
        if not stripped or stripped.startswith("#"):
            continue
        next_line = lines[i + 1].strip()
        if "," in next_line and next_line.split(",", 1)[0] in {"video", "fly"}:
            titles.append(stripped)
    return titles


def _section_table_lines(path: str | Path, section: str) -> list[str]:
    path = Path(path)
    lines = path.read_text().splitlines(keepends=True)
    start_idx = None
    for i, line in enumerate(lines):
        if line.strip().strip('"') == section:
            start_idx = i + 1
            break
    if start_idx is None:
        titles = available_section_titles(path)
        close = _nearby_titles(section, titles)
        detail = f" Available sections include: {close}" if close else ""
        raise ValueError(f"Section not found in {path}: {section!r}.{detail}")

    table_lines: list[str] = []
    for line in lines[start_idx:]:
        if line.strip() == "":
            break
        table_lines.append(line)
    if not table_lines:
        raise ValueError(f"Found section {section!r} in {path}, but it has no table.")
    return table_lines


def _nearby_titles(section: str, titles: list[str], *, limit: int = 8) -> list[str]:
    import difflib

    close = difflib.get_close_matches(section, titles, n=limit, cutoff=0.2)
    if close:
        return close
    return titles[:limit]


def read_learning_stats_section(
    path: str | Path, section: str = DEFAULT_SECTION
) -> pd.DataFrame:
    table_lines = _section_table_lines(path, section)
    return pd.read_csv(StringIO("".join(table_lines)))


def validate_columns(
    df: pd.DataFrame, cols: list[str] | tuple[str, ...], *, context: str
) -> None:
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"{context} is missing requested column(s): {missing}. "
            f"Available columns are: {list(df.columns)}"
        )


def numeric_column(df: pd.DataFrame, col: str) -> tuple[pd.Series, int]:
    raw = df[col]
    numeric = pd.to_numeric(raw, errors="coerce")
    raw_text = raw.astype(str).str.strip()
    non_missing_raw = ~raw_text.str.lower().isin(MISSING_STRINGS)
    dropped = int((non_missing_raw & numeric.isna()).sum())
    return numeric.astype(float), dropped


def parse_group(
    label: str,
    path: str | Path,
    *,
    section: str = DEFAULT_SECTION,
    numeric_cols: list[str] | tuple[str, ...] = DEFAULT_SUMMARY_COLS,
) -> ParsedGroup:
    df = read_learning_stats_section(path, section)
    validate_columns(df, numeric_cols, context=f"Section {section!r} in {path}")
    out = df.copy()
    warnings: dict[str, int] = {}
    for col in numeric_cols:
        out[col], dropped = numeric_column(df, col)
        if dropped:
            warnings[col] = dropped
    return ParsedGroup(label=label, path=str(path), data=out, coercion_warnings=warnings)


def describe_values(values: Any) -> DescriptiveStats:
    x = np.asarray(values, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n == 0:
        return DescriptiveStats(
            n=0,
            mean=np.nan,
            sd=np.nan,
            sem=np.nan,
            ci95_low=np.nan,
            ci95_high=np.nan,
        )
    mean = float(np.mean(x))
    if n == 1:
        return DescriptiveStats(
            n=1,
            mean=mean,
            sd=np.nan,
            sem=np.nan,
            ci95_low=np.nan,
            ci95_high=np.nan,
        )
    sd = float(np.std(x, ddof=1))
    sem = float(sd / np.sqrt(n))
    half = float(stats.t.ppf(0.975, df=n - 1) * sem)
    return DescriptiveStats(
        n=n,
        mean=mean,
        sd=sd,
        sem=sem,
        ci95_low=mean - half,
        ci95_high=mean + half,
    )


def descriptive_rows(
    groups: list[ParsedGroup], summary_cols: list[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in groups:
        validate_columns(group.data, summary_cols, context=f"Group {group.label!r}")
        for col in summary_cols:
            desc = describe_values(group.data[col])
            rows.append(
                {
                    "group": group.label,
                    "column": col,
                    "n": desc.n,
                    "mean": desc.mean,
                    "sd": desc.sd,
                    "sem": desc.sem,
                    "ci95_low": desc.ci95_low,
                    "ci95_high": desc.ci95_high,
                }
            )
    return rows


def paired_test(
    group: ParsedGroup,
    pre_col: str = DEFAULT_PRE_COL,
    post_col: str = DEFAULT_POST_COL,
) -> PairedTest:
    validate_columns(group.data, [pre_col, post_col], context=f"Group {group.label!r}")
    pre = np.asarray(group.data[pre_col], dtype=float)
    post = np.asarray(group.data[post_col], dtype=float)
    keep = np.isfinite(pre) & np.isfinite(post)
    pre = pre[keep]
    post = post[keep]
    reductions = pre - post
    n = int(reductions.size)
    diff_desc = describe_values(reductions)
    if n < 2:
        t_stat = p_value = df = np.nan
    else:
        t_stat, p_value = stats.ttest_rel(pre, post)
        df = float(n - 1)
    return PairedTest(
        group=group.label,
        pre_col=pre_col,
        post_col=post_col,
        n_pairs=n,
        mean_pre=float(np.mean(pre)) if n else np.nan,
        mean_post=float(np.mean(post)) if n else np.nan,
        mean_difference=diff_desc.mean,
        ci95_low=diff_desc.ci95_low,
        ci95_high=diff_desc.ci95_high,
        t_stat=float(t_stat),
        df=float(df),
        p_value=float(p_value),
        reductions=np.asarray(reductions, dtype=float),
    )


def paired_test_rows(tests: list[PairedTest]) -> list[dict[str, Any]]:
    return [
        {
            "group": t.group,
            "pre_col": t.pre_col,
            "post_col": t.post_col,
            "n_pairs": t.n_pairs,
            "mean_pre": t.mean_pre,
            "mean_post": t.mean_post,
            "mean_difference": t.mean_difference,
            "ci95_low": t.ci95_low,
            "ci95_high": t.ci95_high,
            "t_stat": t.t_stat,
            "df": t.df,
            "p_value": t.p_value,
        }
        for t in tests
    ]


def welch_reduction_test(a: PairedTest, b: PairedTest) -> WelchTest:
    xa = np.asarray(a.reductions, dtype=float)
    xb = np.asarray(b.reductions, dtype=float)
    xa = xa[np.isfinite(xa)]
    xb = xb[np.isfinite(xb)]
    n_a = int(xa.size)
    n_b = int(xb.size)
    mean_a = float(np.mean(xa)) if n_a else np.nan
    mean_b = float(np.mean(xb)) if n_b else np.nan
    mean_diff = mean_a - mean_b
    if n_a < 2 or n_b < 2:
        return WelchTest(
            a.group,
            b.group,
            n_a,
            n_b,
            mean_a,
            mean_b,
            mean_diff,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            "Welch independent-samples t-test",
        )

    var_a = float(np.var(xa, ddof=1))
    var_b = float(np.var(xb, ddof=1))
    se2 = var_a / n_a + var_b / n_b
    if se2 <= 0:
        df = t_stat = p_value = ci_low = ci_high = np.nan
    else:
        df = float(
            se2**2
            / ((var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1))
        )
        se = float(np.sqrt(se2))
        t_stat, p_value = stats.ttest_ind(xa, xb, equal_var=False)
        half = float(stats.t.ppf(0.975, df=df) * se)
        ci_low = mean_diff - half
        ci_high = mean_diff + half
    return WelchTest(
        group_a=a.group,
        group_b=b.group,
        n_a=n_a,
        n_b=n_b,
        mean_reduction_a=mean_a,
        mean_reduction_b=mean_b,
        mean_difference_a_minus_b=float(mean_diff),
        ci95_low=float(ci_low),
        ci95_high=float(ci_high),
        t_stat=float(t_stat),
        df=float(df),
        p_value=float(p_value),
        test="Welch independent-samples t-test",
    )


def between_group_rows(test: WelchTest | None) -> list[dict[str, Any]]:
    if test is None:
        return []
    return [
        {
            "group_a": test.group_a,
            "group_b": test.group_b,
            "n_a": test.n_a,
            "n_b": test.n_b,
            "mean_reduction_a": test.mean_reduction_a,
            "mean_reduction_b": test.mean_reduction_b,
            "mean_difference_a_minus_b": test.mean_difference_a_minus_b,
            "ci95_low": test.ci95_low,
            "ci95_high": test.ci95_high,
            "t_stat": test.t_stat,
            "df": test.df,
            "p_value": test.p_value,
            "test": test.test,
        }
    ]


def write_dict_csv(
    path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_wiki_summary_html(
    path: str | Path,
    *,
    section: str,
    summary_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    between_rows: list[dict[str, Any]],
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html>",
                "<head><meta charset=\"utf-8\"><title>Agarose Time Summary</title>",
                "<style>"
                "body{font-family:Arial,sans-serif;line-height:1.35;"
                "max-width:1100px;margin:24px auto;padding:0 16px}"
                "table{border-collapse:collapse;width:100%;margin:12px 0 24px}"
                "th,td{border:1px solid #ccc;padding:5px 7px;text-align:left}"
                "th{background:#f2f2f2}.num{text-align:right}"
                "</style>",
                "</head>",
                "<body>",
                "<h1>Agarose Time Summary</h1>",
                "<p>Summarized percent-time-over-agarose metrics, within-group "
                "paired reductions, and the two-group Welch comparison of paired "
                "reduction scores.</p>",
                f"<p><strong>Metric section analyzed:</strong> {html.escape(section)}</p>",
                "<h2>Descriptive Results</h2>",
                _html_table(summary_rows),
                "<h2>Paired t-tests</h2>",
                _html_table(paired_rows),
                "<h2>Between-genotype Welch Test</h2>",
                (
                    _html_table(between_rows)
                    if between_rows
                    else "<p>Between-group test was skipped because exactly two "
                    "groups were not supplied.</p>"
                ),
                "<h2>Takeaway</h2>",
                f"<p>{html.escape(_takeaway(paired_rows, between_rows))}</p>",
                "</body></html>",
            ]
        )
    )


def _html_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p>No rows.</p>"
    cols = list(rows[0].keys())
    parts = ["<table>", "<thead><tr>"]
    parts.extend(f"<th>{html.escape(col)}</th>" for col in cols)
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        for col in cols:
            val = row.get(col)
            cls = (
                ' class="num"'
                if isinstance(val, (int, float, np.integer, np.floating))
                else ""
            )
            parts.append(f"<td{cls}>{html.escape(_fmt_value(val))}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def _takeaway(
    paired_rows: list[dict[str, Any]], between_rows: list[dict[str, Any]]
) -> str:
    paired_bits = [
        f"{row['group']} showed a mean pre-post reduction of {_fmt_value(row['mean_difference'])} percentage points"
        for row in paired_rows
    ]
    if between_rows:
        row = between_rows[0]
        return (
            "; ".join(paired_bits)
            + f". The between-group reduction difference ({row['group_a']} - {row['group_b']}) "
            f"was {_fmt_value(row['mean_difference_a_minus_b'])} percentage points "
            f"(95% CI {_fmt_value(row['ci95_low'])} to {_fmt_value(row['ci95_high'])}, "
            f"p={_fmt_value(row['p_value'])})."
        )
    return "; ".join(paired_bits) + "."


def _fmt_value(val: Any) -> str:
    if isinstance(val, (float, np.floating)):
        if not np.isfinite(val):
            return "nan"
        return f"{float(val):.6g}"
    return str(val)
