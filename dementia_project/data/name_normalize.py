"""Utilities for mapping filenames to subject names in the metadata CSV."""

from __future__ import annotations

import re

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def normalize_person_name(text: str) -> str:
    """Normalize a person name or filename for robust matching.

    The goal is to make joins robust to punctuation, casing, and separators.
    Example:
        "Robin Williams" -> "robinwilliams"
        "Robin_Williams__01.wav" -> "robinwilliams01wav"
    """
    text = text.strip().casefold()
    text = _NON_ALNUM_RE.sub("", text)
    return text
