"""Accounting consistency checks run after extraction.

An analyst cannot re-read every number, so the pipeline has to say which values
it is unsure about. Two independent signals are used:

* grounding  - the value returned by the model must literally appear in the
               excerpt it cites (catches hallucinated or recomputed numbers);
* consistency - extracted values must agree with each other
               (margin = EBITDA / revenue, guidance low <= high, ...).

Any field with a failed check is routed to the review queue.
"""
from __future__ import annotations

import re


def _digits(text: str) -> str:
    return re.sub(r"[^\d.]", "", str(text))


def is_grounded(raw, context: str) -> bool:
    """True if the number as written by the model appears in the cited excerpt."""
    if raw is None:
        return False
    digits = _digits(raw).strip(".")
    if not digits:  # categorical answer: grounding is checked on keywords instead
        return True
    try:
        value = float(digits)
    except ValueError:
        return False
    # Compare numerically so that "102.0m" matches "£102m" and "1252" matches "1,252".
    context_numbers = {float(m.replace(",", "")) for m in re.findall(r"\d[\d,]*(?:\.\d+)?", context)}
    return value in context_numbers


def consistency_checks(values: dict) -> dict[str, list[str]]:
    """Return {field: [issue, ...]} for fields involved in a failed check."""
    issues: dict[str, list[str]] = {}

    def flag(fields: tuple[str, ...], message: str) -> None:
        for f in fields:
            issues.setdefault(f, []).append(message)

    rev, ebitda, margin = values.get("revenue"), values.get("ebitda"), values.get("ebitda_margin_pct")
    if rev and ebitda is not None and margin is not None:
        implied = ebitda / rev * 100
        if abs(implied - margin) > 0.6:
            flag(("revenue", "ebitda", "ebitda_margin_pct"),
                 f"margin {margin:.1f}% inconsistent with EBITDA/revenue = {implied:.1f}%")

    lev = values.get("leverage_ratio")
    if lev is not None and not 0 <= lev <= 15:
        flag(("leverage_ratio",), f"leverage {lev}x outside plausible range")

    g = values.get("revenue_growth_pct")
    if g is not None and abs(g) > 150:
        flag(("revenue_growth_pct",), f"growth {g}% implausible")

    lo, hi = values.get("guidance_low"), values.get("guidance_high")
    if lo is not None and hi is not None and lo > hi:
        flag(("guidance_low", "guidance_high"), "guidance low end above high end")
    if rev and lo is not None and not 2.5 * rev <= lo <= 5.5 * rev:
        flag(("guidance_low", "revenue"), "full-year guidance not in line with quarterly revenue (unit error?)")

    for name in ("capex", "cash"):
        v = values.get(name)
        if rev and v is not None and v > 3 * rev:
            flag((name,), f"{name} larger than 3x quarterly revenue (unit error?)")

    # Net debt / leverage must imply an LTM EBITDA close to 4x the quarterly EBITDA.
    nd = values.get("net_debt")
    if nd is not None and lev and ebitda:
        implied_ltm = nd / lev
        if not 0.6 <= implied_ltm / (4 * ebitda) <= 1.6:
            flag(("net_debt", "leverage_ratio"),
                 f"net debt / leverage implies LTM EBITDA of {implied_ltm:.1f}, vs ~{4 * ebitda:.1f} from the quarter")
    return issues


def type_checks(raws: dict, kinds: dict) -> dict[str, list[str]]:
    """Catch values of the wrong kind, e.g. a leverage multiple returned for an amount."""
    issues: dict[str, list[str]] = {}
    for name, raw in raws.items():
        if raw is None:
            continue
        text = str(raw).strip().lower()
        if kinds[name] == "money" and (text.endswith("x") or text.endswith("%")):
            issues.setdefault(name, []).append(f"'{raw}' looks like a ratio or percentage, not an amount")
    return issues


def review_issues(fields: dict, kinds: dict) -> dict[str, list[str]]:
    """All post-extraction checks. ``fields`` maps name -> {"value": ..., "raw": ...}."""
    numeric = {k: f["value"] for k, f in fields.items()
               if isinstance(f["value"], (int, float)) and not isinstance(f["value"], bool)}
    issues = type_checks({k: f["raw"] for k, f in fields.items()}, kinds)
    for name, problems in consistency_checks(numeric).items():
        issues.setdefault(name, []).extend(problems)
    return issues
