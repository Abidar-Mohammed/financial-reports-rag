"""The extraction schema: which KPIs are pulled from each report and how they are searched for.

Fields are processed in small groups. Each group gets its own retrieval and its
own LLM call, which keeps prompts short (important for small models) while
avoiding one call per field.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FieldSpec:
    name: str
    kind: str  # "money" (normalised to millions), "pct", "ratio", "count", "category"
    group: str
    description: str  # shown to the LLM
    query: str  # used for retrieval


FIELDS: list[FieldSpec] = [
    FieldSpec("revenue", "money", "pnl", "revenue / net sales / turnover for the quarter",
              "quarterly revenue net sales turnover"),
    FieldSpec("revenue_growth_pct", "pct", "pnl", "revenue growth versus the same quarter of the prior year, in %",
              "revenue growth versus prior year percentage"),
    FieldSpec("ebitda", "money", "pnl", "adjusted / recurring EBITDA for the quarter",
              "adjusted EBITDA for the quarter"),
    FieldSpec("ebitda_margin_pct", "pct", "pnl", "EBITDA margin for the quarter, in %",
              "EBITDA margin percentage"),
    FieldSpec("net_debt", "money", "balance", "net debt / net financial debt at the end of the period",
              "net debt net financial debt end of period balance sheet"),
    FieldSpec("cash", "money", "balance", "cash and cash equivalents at the end of the period",
              "cash and cash equivalents liquidity"),
    FieldSpec("leverage_ratio", "ratio", "balance", "leverage ratio (net debt / LTM EBITDA), e.g. 3.2x",
              "leverage ratio net debt to LTM EBITDA multiple"),
    FieldSpec("capex", "money", "operations", "capital expenditure for the quarter",
              "capital expenditure capex investment"),
    FieldSpec("headcount", "count", "operations", "number of employees / FTEs at the end of the quarter",
              "number of employees full-time equivalents FTE headcount"),
    FieldSpec("covenant_status", "category", "operations",
              "status of the financial covenant test: one of compliant, breach, waiver",
              "financial covenant compliance breach waiver lenders"),
    FieldSpec("guidance_low", "money", "outlook", "low end of the full-year revenue guidance range",
              "full year revenue guidance range outlook"),
    FieldSpec("guidance_high", "money", "outlook", "high end of the full-year revenue guidance range",
              "full year revenue guidance range outlook"),
]

FIELD_BY_NAME = {f.name: f for f in FIELDS}
GROUPS: dict[str, list[FieldSpec]] = {}
for _f in FIELDS:
    GROUPS.setdefault(_f.group, []).append(_f)
