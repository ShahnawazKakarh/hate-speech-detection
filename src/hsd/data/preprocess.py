"""Text preprocessing utilities.

Light-touch normalization shared across all three models. Each model can apply
additional model-specific preprocessing (e.g., tokenization for transformers).
"""

from __future__ import annotations

import re

import emoji

# Patterns
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
RT_RE = re.compile(r"^RT\s+", flags=re.IGNORECASE)
MULTI_WS_RE = re.compile(r"\s+")
HTML_ENTITY_RE = re.compile(r"&[a-z]+;|&#\d+;")


def clean_text(
    text: str,
    *,
    lowercase: bool = True,
    strip_urls: bool = True,
    strip_mentions: bool = True,
    keep_hashtag_text: bool = True,
    demojize: bool = True,
) -> str:
    """Normalize a single comment.

    The defaults mirror common practice for Twitter-style hate-speech corpora:
    strip URLs/mentions, keep hashtag text (the word often carries the signal),
    convert emojis to text tokens, collapse whitespace.
    """
    if not isinstance(text, str):
        return ""

    text = RT_RE.sub("", text)
    text = HTML_ENTITY_RE.sub(" ", text)

    if strip_urls:
        text = URL_RE.sub(" ", text)
    if strip_mentions:
        text = MENTION_RE.sub(" ", text)

    if keep_hashtag_text:
        text = HASHTAG_RE.sub(r"\1", text)
    else:
        text = HASHTAG_RE.sub(" ", text)

    if demojize:
        text = emoji.demojize(text, delimiters=(" ", " "))

    if lowercase:
        text = text.lower()

    text = MULTI_WS_RE.sub(" ", text).strip()
    return text


def clean_series(texts):
    """Vectorized cleaning over an iterable of strings."""
    return [clean_text(t) for t in texts]
