import pytest

from src.utils.parsers import decode_label_newlines, normalize_multiline_label


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (r"Control\ncondition", "Control\ncondition"),
        (r"Control\\ncondition", r"Control\ncondition"),
        (r"Control\tcondition", r"Control\tcondition"),
        (r"Control\condition", r"Control\condition"),
        (
            r"Control\ncondition|Experimental\ntreatment",
            "Control\ncondition|Experimental\ntreatment",
        ),
        ("Control\ncondition", "Control\ncondition"),
    ],
)
def test_decode_label_newlines(raw, expected):
    assert decode_label_newlines(raw) == expected


def test_decode_label_newlines_uses_backslash_pairing():
    assert decode_label_newlines("Label" + "\\" * 3 + "nnext") == "Label\\\nnext"


def test_normalize_multiline_label_preserves_explicit_breaks():
    label = "  Long   control label  \n  with   treatment  "

    assert normalize_multiline_label(label) == "Long control label\nwith treatment"
