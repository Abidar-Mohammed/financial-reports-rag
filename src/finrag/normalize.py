"""Turn raw strings returned by the LLM into comparable numbers.

The LLM is asked to copy values *as written* and to cite the excerpt it used.
Unit scaling is then resolved in code from the cited excerpt (a table caption
saying EUR '000, a "k" suffix, the word "million"...). Keeping arithmetic out of
the model is cheaper and more reliable than asking it to convert units.
"""
from __future__ import annotations

import re

NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
THOUSANDS_RE = re.compile(r"'000|\b(?:thousands?|k)\b|\d\s*k\b", re.IGNORECASE)
MILLIONS_RE = re.compile(r"\b(?:millions?|mn|m)\b|\d\s*m\b|\b[A-Z]{3}\s*m\b", re.IGNORECASE)
CATEGORY_ALIASES = {
    "compliant": "compliant", "compliance": "compliant", "in compliance": "compliant", "met": "compliant",
    "breach": "breach", "breached": "breach", "not compliant": "breach",
    "waiver": "waiver", "waived": "waiver",
}


def parse_number(raw: str | float | int | None) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    match = NUMBER_RE.search(str(raw).replace("−", "-"))
    if not match:
        return None
    return float(match.group(0).replace(",", ""))


def scale_of(raw: str, context: str) -> float:
    """Multiplier that converts the raw value into millions."""
    raw_s = str(raw)
    if THOUSANDS_RE.search(raw_s):
        return 1e-3
    if MILLIONS_RE.search(raw_s):
        return 1.0
    # Fall back to the cited excerpt. Table captions carry the unit for the whole table.
    if THOUSANDS_RE.search(context):
        return 1e-3
    return 1.0


def normalize_value(kind: str, raw, context: str = ""):
    if raw is None or (isinstance(raw, str) and raw.strip().lower() in {"", "null", "none", "n/a", "not found"}):
        return None
    if kind == "category":
        text = str(raw).strip().lower()
        for alias, value in CATEGORY_ALIASES.items():
            if alias in text:
                return value
        return None
    number = parse_number(raw)
    if number is None:
        return None
    if kind == "money":
        return round(number * scale_of(raw, context), 6)
    if kind == "count":
        return int(round(number))
    return number  # pct and ratio are used as written


def is_correct(kind: str, predicted, truth, rel_tol: float = 0.005) -> bool:
    if predicted is None:
        return False
    if kind == "category":
        return predicted == truth
    if kind in {"pct", "ratio"}:
        return abs(float(predicted) - float(truth)) <= 0.051
    if kind == "count":
        return int(predicted) == int(truth)
    return abs(float(predicted) - float(truth)) <= rel_tol * max(abs(float(truth)), 1e-9)
