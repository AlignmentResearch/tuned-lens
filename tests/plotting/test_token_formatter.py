import pytest
from tuned_lens.plotting import TokenFormatter


@pytest.fixture
def formatter():
    return TokenFormatter()


def test_format_non_string(formatter):
    assert formatter.format(None) == "<unk>"
    assert formatter.format(123) == "<unk>"


def test_format_ellipsis(formatter):
    token = "ThisIsALongToken"
    expected = "ThisIs…"
    assert formatter.format(token) == expected


def test_format_no_ellipsis(formatter):
    formatter.max_string_len = None
    token = "ThisIsALongToken"
    expected = "ThisIsALongToken"
    assert formatter.format(token) == expected


def test_format_newline_token_replacement(formatter):
    formatter.max_string_len = None
    token = "HelloĊWorld"
    expected = "Hello\\nWorld"
    assert formatter.format(token) == expected


def test_format_whitespace_token_replacement(formatter):
    formatter.max_string_len = None
    token = "HelloĠWorld"
    expected = "Hello_World"
    assert formatter.format(token) == expected


def test_format_multiple_replacements(formatter):
    formatter.max_string_len = None
    token = "Line1ĊLine2ĠLine3"
    expected = "Line1\\nLine2_Line3"
    assert formatter.format(token) == expected
