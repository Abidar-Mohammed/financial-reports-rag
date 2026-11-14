"""Unit tests for the deterministic parts of the pipeline (no model download needed)."""
from pathlib import Path

import pytest

from finrag.extraction import _parse_json
from finrag.ingest import parse_pdf
from finrag.normalize import is_correct, normalize_value
from finrag.validation import consistency_checks, is_grounded, review_issues

REPORTS = Path(__file__).resolve().parents[1] / "data" / "reports"


@pytest.mark.parametrize("raw, context, expected", [
    ("18,388", "Key figures for the quarter (EUR '000).", 18.388),
    ("48.3", "Key figures for the quarter (EUR m).", 48.3),
    ("EUR 713k", "", 0.713),
    ("€3.4 million", "", 3.4),
    ("€73m", "Key figures (EUR '000)", 73.0),  # explicit unit beats the table caption
    (None, "", None),
    ("n/a", "", None),
])
def test_money_normalisation(raw, context, expected):
    assert normalize_value("money", raw, context) == expected


def test_other_kinds():
    assert normalize_value("pct", "33.2%") == 33.2
    assert normalize_value("ratio", "7.5x") == 7.5
    assert normalize_value("count", "1,240") == 1240
    assert normalize_value("category", "The covenant was breached") == "breach"
    assert normalize_value("category", "Compliant") == "compliant"


def test_scoring_tolerances():
    assert is_correct("money", 18.39, 18.388)
    assert not is_correct("money", 18.9, 18.388)
    assert is_correct("pct", 10.5, 10.5)
    assert not is_correct("count", 336, 335)
    assert not is_correct("money", None, 1.0)


def test_grounding():
    ctx = "Net debt | 59,682 | 62,392"
    assert is_grounded("59,682", ctx)
    assert is_grounded("59682", ctx)
    assert not is_grounded("59,700", ctx)
    assert is_grounded("102.0m", "a range of £102m to £106m")


def test_consistency_checks_flag_unit_errors():
    ok = {"revenue": 18.388, "ebitda": 1.928, "ebitda_margin_pct": 10.5, "guidance_low": 73, "guidance_high": 77}
    assert consistency_checks(ok) == {}
    bad = dict(ok, ebitda=1928.0)  # thousands not converted
    assert "ebitda" in consistency_checks(bad)
    swapped = dict(ok, guidance_low=77, guidance_high=73)
    assert "guidance_low" in consistency_checks(swapped)


def test_leverage_returned_as_net_debt_is_flagged():
    # Real failure seen in evaluation: the model returned the leverage multiple for net debt.
    fields = {
        "revenue": {"value": 26.0, "raw": "26.0"}, "ebitda": {"value": 7.0, "raw": "7.0"},
        "leverage_ratio": {"value": 1.7, "raw": "1.7x"}, "net_debt": {"value": 1.7, "raw": "1.7x"},
    }
    kinds = {"revenue": "money", "ebitda": "money", "leverage_ratio": "ratio", "net_debt": "money"}
    issues = review_issues(fields, kinds)
    assert "net_debt" in issues
    assert any("ratio" in i for i in issues["net_debt"])


def test_parse_json_tolerates_noise():
    assert _parse_json('Sure:\n```json\n{"a": {"value": "1"}}\n```') == {"a": {"value": "1"}}
    assert _parse_json("no json here") == {}


@pytest.mark.skipif(not REPORTS.exists(), reason="run scripts/generate_reports.py first")
def test_tables_are_kept_whole():
    chunks = parse_pdf(REPORTS / "helios_e_mobility_q3_2025.pdf")
    tables = [c for c in chunks if c.kind == "table"]
    assert len(tables) == 2
    assert "EUR '000" in tables[0].text  # caption carrying the unit stays with the table
    assert tables[0].text.count("\n") >= 4  # header + rows
    assert all(c.company == "Helios E-Mobility" and c.period == "Q3 2025" for c in chunks)
