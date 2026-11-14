"""Rule-based baseline: keyword lookup + first number after the label.

This is what a first version of such a tool typically looks like (and what many
analysts do with Ctrl+F). It is used as the reference point in the evaluation.
"""
from __future__ import annotations

import re

from .ingest import Chunk
from .normalize import normalize_value
from .schema import FIELDS

LABELS = {
    "revenue": ["Revenue", "Net sales", "Net revenue"],
    "revenue_growth_pct": ["Growth vs. prior year"],
    "ebitda": ["Adjusted EBITDA", "EBITDA (adj.)"],
    "ebitda_margin_pct": ["Adjusted EBITDA margin", "EBITDA (adj.) margin"],
    "net_debt": ["Net debt", "Net financial debt"],
    "cash": ["Cash and cash equivalents"],
    "leverage_ratio": ["Leverage (net debt / LTM EBITDA)"],
    "capex": ["Capital expenditure"],
    "headcount": ["employed"],
    "covenant_status": ["covenant"],
    "guidance_low": ["guidance to a range of"],
    "guidance_high": ["m to"],
}
NUMBER = r"[^\d\-]{0,25}?(-?\d[\d,]*(?:\.\d+)?\s*(?:%|x|k\b|m\b|million)?)"


def extract_baseline(chunks: list[Chunk]) -> dict:
    """Return {field: value} for one document."""
    tables = "\n".join(c.text for c in chunks if c.kind == "table")
    text = " ".join(c.text for c in chunks if c.kind == "text")
    full = tables + "\n" + text
    out = {}
    for spec in FIELDS:
        value = None
        if spec.name == "covenant_status":
            low = full.lower()
            value = "breach" if "breached" in low else "waiver" if "waiver" in low else "compliant" if "compliance" in low else None
        else:
            for label in LABELS[spec.name]:
                m = re.search(re.escape(label) + NUMBER, full)
                if m:
                    # the caption of the first table gives the unit of table figures
                    context = tables if m.start() < len(tables) else m.group(0)
                    value = normalize_value(spec.kind, m.group(1), context)
                    break
        out[spec.name] = value
    return out
