"""Tests for text preprocessing."""

from hsd.data.preprocess import clean_text


def test_clean_text_strips_url():
    out = clean_text("check this https://example.com out")
    assert "http" not in out
    assert "example" not in out


def test_clean_text_strips_mention():
    out = clean_text("hey @user how are you")
    assert "@user" not in out


def test_clean_text_keeps_hashtag_word():
    out = clean_text("#freedom is great")
    assert "freedom" in out
    assert "#" not in out


def test_clean_text_lowercases():
    assert clean_text("HELLO World") == "hello world"


def test_clean_text_handles_non_string():
    assert clean_text(None) == ""
    assert clean_text(123) == ""


def test_clean_text_collapses_whitespace():
    assert clean_text("a   b\t\tc\n\nd") == "a b c d"


def test_clean_text_strips_rt():
    assert not clean_text("RT @x hello world").startswith("rt")
