"""Generate a synthetic corpus of quarterly portfolio-company reports.

Real portfolio reports are confidential, so the evaluation corpus is generated.
The generator deliberately reproduces the things that make these documents hard
to process in practice:

* the same metric appears under different names (Revenue / Net sales / Turnover)
* figures are reported in millions for some companies and thousands for others
* tables carry distractor columns (prior year, budget) next to the actual figure
* some KPIs only appear in narrative text, never in a table
* currencies differ across companies (EUR, USD, GBP)

Every generated value is stored in ``data/ground_truth.json`` together with the
exact string printed in the PDF, which is what the evaluation uses as evidence.

Usage:
    python scripts/generate_reports.py --out data --seed 7
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

QUARTERS = ["Q1", "Q2", "Q3", "Q4"]
YEAR = 2025
SYMBOL = {"EUR": "€", "USD": "$", "GBP": "£"}


@dataclass
class Company:
    name: str
    sector: str
    activity: str
    currency: str
    unit: str  # "m" or "k": unit used in tables
    base_revenue: float  # quarterly revenue in millions, Q1
    growth: float  # YoY growth, %
    margin: float  # EBITDA margin, %
    net_debt: float  # millions
    fte: int
    ceo: str
    revenue_label: str
    ebitda_label: str
    net_debt_label: str
    style: str = "A"  # report template; "B" is held out and only used for the robustness test


# Wording that differs between the two report templates.
STYLES = {
    "A": {
        "growth": "Growth vs. prior year", "margin": "{ebitda} margin", "cash": "Cash and cash equivalents",
        "leverage": "Leverage (net debt / LTM EBITDA)", "capex_row": "Capital expenditure",
        "prior_first": False,
        "headcount": "At the end of the quarter the group employed {n} full-time equivalents (FTEs). ",
        "capex": "Capital expenditure for the quarter amounted to {v}, mainly related to capacity extensions and the "
                 "replacement of end-of-life equipment.",
        "guidance": "For the full year {y}, management {verb} its revenue guidance to a range of {lo} to {hi}. ",
    },
    "B": {
        "growth": "Change y-o-y", "margin": "Operating margin", "cash": "Liquid funds",
        "leverage": "Net leverage ratio", "capex_row": "Investments in fixed assets",
        "prior_first": True,  # prior-year column printed before the actual column
        "headcount": "Our workforce totalled {n} people on a full-time basis at period end. ",
        "capex": "Investments in fixed assets totalled {v} in the quarter, of which the majority was growth related.",
        "guidance": "Looking at {y} as a whole, the board {verb} that sales should land between {lo} and {hi}. ",
    },
}


COMPANIES = [
    Company("Valtera Automotive Components", "Automotive", "a Tier-2 supplier of brake and steering sub-assemblies for European OEMs",
            "EUR", "m", 52.0, 6.5, 14.2, 118.0, 1240, "Martin Kessler", "Revenue", "Adjusted EBITDA", "Net financial debt"),
    Company("Helios E-Mobility", "Automotive / EV infrastructure", "a developer and operator of fast-charging hubs for electric vehicles",
            "EUR", "k", 18.5, 34.0, 9.5, 64.0, 310, "Ines Moreau", "Net sales", "EBITDA (adj.)", "Net debt"),
    Company("Norbridge Logistics", "Transport & Logistics", "a contract logistics provider for retail and e-commerce clients",
            "EUR", "m", 71.0, 4.1, 8.7, 142.0, 2870, "Lars Henningsen", "Turnover", "Recurring EBITDA", "Net debt"),
    Company("Clarion Diagnostics", "Healthcare", "a network of clinical laboratories offering routine and specialty testing",
            "USD", "k", 39.0, 9.8, 22.5, 155.0, 960, "Dana Whitfield", "Revenue", "Adjusted EBITDA", "Net debt"),
    Company("Fenwick Payments", "Financial services", "a payment processing platform for small and mid-sized merchants",
            "GBP", "m", 26.0, 18.0, 27.0, 48.0, 420, "Oliver Grant", "Net revenue", "Adjusted EBITDA", "Net financial debt"),
    # Held-out template (style "B"): different wording and column order.
    Company("Brenner Fahrzeugtechnik", "Automotive", "a manufacturer of thermal management modules for electric and hybrid vehicles",
            "EUR", "m", 44.0, 7.5, 12.0, 96.0, 870, "Katrin Albers", "Sales", "Operating EBITDA", "Net borrowings", "B"),
    Company("Corvane Software", "Software", "a provider of fleet management software for leasing companies and dealerships",
            "USD", "k", 12.5, 21.0, 31.0, 38.0, 185, "Ravi Menon", "Recurring revenue", "Operating EBITDA", "Net borrowings", "B"),
]

FILLER = [
    "Market conditions remained broadly in line with the assumptions of the investment case. "
    "Management continued to focus on commercial discipline, pricing and the integration of recent initiatives.",
    "The board met twice during the quarter and reviewed the strategic plan, the pipeline of add-on acquisitions "
    "and the progress of the operational improvement programme agreed at closing.",
    "Input cost inflation continued to moderate, although wage pressure in several sites remained a point of attention. "
    "Price increases negotiated with key customers are expected to offset most of the impact over the next two quarters.",
    "The ESG roadmap progressed as planned, with the publication of the first carbon footprint assessment "
    "and the roll-out of a group-wide health and safety reporting tool.",
    "Working capital management remained a priority. Days sales outstanding improved slightly thanks to "
    "tighter collection procedures, while inventory levels were kept close to target.",
]

RISKS = [
    "customer concentration, with the top three customers representing a significant share of sales",
    "execution risk on the ongoing ERP migration",
    "exposure to energy prices at the main production sites",
    "tight labour market for qualified technicians",
    "pressure on pricing from larger competitors",
    "regulatory changes affecting reimbursement or pricing",
    "integration of the most recent add-on acquisition",
]


def fmt_amount(value_m: float, unit: str) -> str:
    """Format a value expressed in millions the way it appears in a table."""
    if unit == "m":
        return f"{value_m:,.1f}"
    return f"{value_m * 1000:,.0f}"


def prose_amount(value_m: float, currency: str, unit: str) -> str:
    """Format a value expressed in millions the way it appears in narrative text."""
    sym = SYMBOL[currency]
    if unit == "m":
        return f"{sym}{value_m:,.1f} million"
    return f"{currency} {value_m * 1000:,.0f}k"


def build_report(company: Company, q_index: int, rng: random.Random) -> tuple[list, dict]:
    quarter = QUARTERS[q_index]
    period = f"{quarter} {YEAR}"

    # ---- figures (all monetary values in millions) -----------------------
    # Companies reporting in thousands get thousand-level precision.
    p = 1 if company.unit == "m" else 3
    seasonality = [0.0, 0.03, -0.02, 0.05][q_index]
    revenue = round(company.base_revenue * (1 + 0.015 * q_index + seasonality) * rng.uniform(0.98, 1.02), p)
    growth = round(company.growth + rng.uniform(-2.5, 2.5), 1)
    revenue_py = round(revenue / (1 + growth / 100), p)
    growth = round((revenue / revenue_py - 1) * 100, 1)  # keep consistent with rounded figures
    budget = round(revenue * rng.uniform(0.96, 1.06), p)

    margin_target = company.margin + rng.uniform(-1.5, 1.5)
    ebitda = round(revenue * margin_target / 100, p)
    margin = round(ebitda / revenue * 100, 1)
    ebitda_py = round(revenue_py * (company.margin + rng.uniform(-2, 1)) / 100, p)

    net_debt = round(company.net_debt * (1 - 0.02 * q_index) * rng.uniform(0.97, 1.03), p)
    ltm_ebitda = round(ebitda * 4 * rng.uniform(0.95, 1.03), p)
    leverage = round(net_debt / ltm_ebitda, 1)
    cash = round(revenue * rng.uniform(0.25, 0.6), p)
    capex = round(revenue * rng.uniform(0.03, 0.09), p)
    headcount = int(company.fte * (1 + 0.01 * q_index) + rng.randint(-15, 25))

    covenant_threshold = round(leverage + rng.choice([0.6, 0.8, 1.0, 1.2]), 1)
    covenant_status = "compliant"
    if company.name.startswith("Helios") and quarter == "Q3":
        covenant_threshold = round(leverage - 0.3, 1)
        covenant_status = "breach"
    if company.name.startswith("Helios") and quarter == "Q4":
        covenant_status = "waiver"

    annual = revenue * 4 * rng.uniform(0.98, 1.03)
    guidance_low = round(annual * 0.98, 0)
    guidance_high = guidance_low + rng.choice([5, 10, 15]) if annual > 150 else guidance_low + rng.choice([2, 4, 5])

    cur, unit = company.currency, company.unit
    unit_label = f"{cur} m" if unit == "m" else f"{cur} '000"

    # ---- layout ---------------------------------------------------------------
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=16, spaceAfter=4)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=9.5, leading=13, alignment=TA_LEFT)
    small = ParagraphStyle("small", parent=body, fontSize=8, textColor=colors.grey)

    def table(rows: list[list[str]]) -> Table:
        t = Table(rows, hAlign="LEFT", colWidths=[6.2 * cm] + [3.0 * cm] * (len(rows[0]) - 1))
        t.setStyle(TableStyle([
            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.5),
            ("FONT", (0, 1), (-1, -1), "Helvetica", 8.5),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8ECF1")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9AA5B1")),
            ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ]))
        return t

    capex_in_table = rng.random() < 0.5
    story: list = []
    story += [
        Paragraph(company.name, h1),
        Paragraph(f"Quarterly Portfolio Report &mdash; {period}", h2),
        Paragraph(f"Sector: {company.sector}. Reporting currency: {cur}. Prepared by management for the investment committee. "
                  "Unaudited figures.", small),
        Spacer(1, 8),
        Paragraph("1. Executive summary", h2),
        Paragraph(
            f"{company.name} is {company.activity}. In {period} the company generated "
            f"{company.revenue_label.lower()} of {prose_amount(revenue, cur, unit)}, "
            f"{'up' if growth >= 0 else 'down'} {abs(growth):.1f}% compared with the same quarter of last year. "
            f"{company.ebitda_label} reached {prose_amount(ebitda, cur, unit)}, corresponding to a margin of {margin:.1f}%.", body),
        Paragraph(rng.choice(FILLER), body),
        Paragraph("2. Financial performance", h2),
        Paragraph(f"Key figures for the quarter ({unit_label}).", small),
    ]
    style = STYLES[company.style]
    perf_rows = [
        ["Metric", period, f"{quarter} {YEAR - 1}", "Budget"],
        [company.revenue_label, fmt_amount(revenue, unit), fmt_amount(revenue_py, unit), fmt_amount(budget, unit)],
        [style["growth"], f"{growth:.1f}%", "n/a", "n/a"],
        [company.ebitda_label, fmt_amount(ebitda, unit), fmt_amount(ebitda_py, unit), fmt_amount(budget * company.margin / 100, unit)],
        [style["margin"].format(ebitda=company.ebitda_label), f"{margin:.1f}%", f"{ebitda_py / revenue_py * 100:.1f}%",
         f"{company.margin:.1f}%"],
    ]
    if capex_in_table:
        perf_rows.append([style["capex_row"], fmt_amount(capex, unit), fmt_amount(capex * rng.uniform(0.8, 1.2), unit),
                          fmt_amount(capex * rng.uniform(0.9, 1.3), unit)])
    if style["prior_first"]:  # swap the actual and prior-year columns
        perf_rows = [[r[0], r[2], r[1], r[3]] for r in perf_rows]
    story += [table(perf_rows), Spacer(1, 6)]
    story.append(Paragraph(
        f"Year-to-date {company.revenue_label.lower()} amounted to "
        f"{prose_amount(revenue * (q_index + 1) * rng.uniform(0.96, 1.0), cur, unit)}. "
        + rng.choice(FILLER), body))

    previous = f"End of {QUARTERS[q_index - 1] if q_index else 'Q4'} {YEAR if q_index else YEAR - 1}"
    balance_rows = [
        ["Item", f"End of {period}", previous],
        [style["cash"], fmt_amount(cash, unit), fmt_amount(cash * rng.uniform(0.8, 1.2), unit)],
        [company.net_debt_label, fmt_amount(net_debt, unit), fmt_amount(net_debt * rng.uniform(1.0, 1.05), unit)],
        ["LTM EBITDA", fmt_amount(ltm_ebitda, unit), fmt_amount(ltm_ebitda * rng.uniform(0.95, 1.0), unit)],
        [style["leverage"], f"{leverage:.1f}x", f"{leverage * rng.uniform(1.0, 1.1):.1f}x"],
    ]
    if style["prior_first"]:
        balance_rows = [[r[0], r[2], r[1]] for r in balance_rows]
    story += [
        Paragraph("3. Balance sheet and liquidity", h2),
        Paragraph(f"Balance sheet items at the end of the period ({unit_label}).", small),
        table(balance_rows),
        Spacer(1, 6),
        Paragraph("Liquidity remains adequate, supported by an undrawn revolving credit facility. "
                  "No dividend or shareholder distribution was made during the period.", body),
        PageBreak(),
        Paragraph("4. Operations", h2),
        Paragraph(style["headcount"].format(n=f"{headcount:,}") + rng.choice(FILLER), body),
    ]
    if not capex_in_table:
        story.append(Paragraph(style["capex"].format(v=prose_amount(capex, cur, unit)), body))
    else:
        story.append(Paragraph("Investment projects progressed in line with the capital expenditure plan.", body))

    covenant_text = {
        "compliant": f"The company was in compliance with its financial covenants at the test date. The leverage covenant "
                     f"is set at {covenant_threshold:.1f}x, compared with an actual leverage of {leverage:.1f}x.",
        "breach": f"The leverage covenant of {covenant_threshold:.1f}x was breached at the test date, with actual leverage of "
                  f"{leverage:.1f}x. Management has opened discussions with the lenders.",
        "waiver": "Following the breach reported last quarter, the lenders granted a covenant waiver and reset the leverage "
                  f"test to {covenant_threshold:.1f}x. Actual leverage stood at {leverage:.1f}x.",
    }[covenant_status]
    story += [
        Paragraph("5. Financing and covenants", h2),
        Paragraph(covenant_text, body),
        Paragraph("6. Outlook", h2),
        Paragraph(
            style["guidance"].format(y=YEAR, verb="confirms" if rng.random() < 0.6 else "updates" if company.style == "A" else "expects",
                                     lo=f"{SYMBOL[cur]}{guidance_low:,.0f}m", hi=f"{SYMBOL[cur]}{guidance_high:,.0f}m")
            + f"The medium-term plan presented at closing assumed revenue of {SYMBOL[cur]}{guidance_high * 1.3:,.0f}m by {YEAR + 2}.", body),
        Paragraph("7. Key risks", h2),
        Paragraph("The main risks monitored by management are " + "; ".join(rng.sample(RISKS, 3)) + ".", body),
        Spacer(1, 10),
        Paragraph(f"{company.ceo}, Chief Executive Officer", small),
    ]

    truth = {
        "company": company.name,
        "period": period,
        "currency": cur,
        "fields": {
            "revenue": revenue,
            "revenue_growth_pct": growth,
            "ebitda": ebitda,
            "ebitda_margin_pct": margin,
            "net_debt": net_debt,
            "cash": cash,
            "leverage_ratio": leverage,
            "capex": capex,
            "headcount": headcount,
            "covenant_status": covenant_status,
            "guidance_low": guidance_low,
            "guidance_high": guidance_high,
        },
        # The string printed in the PDF for each field: used as retrieval evidence.
        "evidence": {
            "revenue": fmt_amount(revenue, unit),
            "revenue_growth_pct": f"{growth:.1f}%",
            "ebitda": fmt_amount(ebitda, unit),
            "ebitda_margin_pct": f"{margin:.1f}%",
            "net_debt": fmt_amount(net_debt, unit),
            "cash": fmt_amount(cash, unit),
            "leverage_ratio": f"{leverage:.1f}x",
            "capex": fmt_amount(capex, unit) if capex_in_table else prose_amount(capex, cur, unit),
            "headcount": f"{headcount:,}",
            "covenant_status": {"compliant": "in compliance", "breach": "was breached", "waiver": "covenant waiver"}[covenant_status],
            "guidance_low": f"{SYMBOL[cur]}{guidance_low:,.0f}m",
            "guidance_high": f"{SYMBOL[cur]}{guidance_high:,.0f}m",
        },
        "difficulty": {"unit": unit, "capex_location": "table" if capex_in_table else "text", "template": company.style},
    }
    return story, truth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data", type=Path)
    parser.add_argument("--seed", default=7, type=int)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    reports_dir = args.out / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    ground_truth = {}
    for company in COMPANIES:
        for q in range(4):
            story, truth = build_report(company, q, rng)
            slug = company.name.lower().replace(" ", "_").replace("-", "_")
            filename = f"{slug}_{QUARTERS[q].lower()}_{YEAR}.pdf"
            doc = SimpleDocTemplate(str(reports_dir / filename), pagesize=A4, title=f"{company.name} {truth['period']}",
                                    leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm)
            doc.build(story)
            ground_truth[filename] = truth

    (args.out / "ground_truth.json").write_text(json.dumps(ground_truth, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Generated {len(ground_truth)} reports in {reports_dir}")


if __name__ == "__main__":
    main()
