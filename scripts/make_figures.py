"""Produce the figures used in the README from results/*.json.

    python scripts/make_figures.py --extractions results/extractions_qwen2.5-3b.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pymupdf  # noqa: E402

OUT = Path("docs/figures")
INK, MUTED, GRID = "#1F2933", "#7B8794", "#E4E7EB"
C_MAIN, C_ALT, C_WARN = "#2F5D8A", "#B8C4D0", "#C26D3A"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True, "figure.dpi": 150,
})

PRETTY = {
    "bm25": "BM25", "dense": "Dense (bge-small)", "hybrid": "Hybrid (RRF)", "hybrid_rerank": "Hybrid + rerank",
    "revenue": "Revenue", "revenue_growth_pct": "Revenue growth %", "ebitda": "EBITDA", "ebitda_margin_pct": "EBITDA margin %",
    "net_debt": "Net debt", "cash": "Cash", "leverage_ratio": "Leverage (x)", "capex": "Capex", "headcount": "Headcount",
    "covenant_status": "Covenant status", "guidance_low": "Guidance (low)", "guidance_high": "Guidance (high)",
}


def _label_bars(ax, bars, fmt="{:.0%}", horizontal=False):
    for b in bars:
        v = b.get_width() if horizontal else b.get_height()
        if horizontal:
            ax.text(v + 0.01, b.get_y() + b.get_height() / 2, fmt.format(v), va="center", fontsize=8, color=INK)
        else:
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, fmt.format(v), ha="center", fontsize=8, color=INK)


def retrieval_figure(metrics: dict) -> None:
    modes = list(metrics)
    ks = ["hit@1", "hit@3", "hit@5"]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    width = 0.2
    shades = ["#C9D3DD", "#8FA3B8", "#5B7A99", C_MAIN]
    for i, mode in enumerate(modes):
        xs = [j + (i - 1.5) * width for j in range(len(ks))]
        bars = ax.bar(xs, [metrics[mode][k] for k in ks], width, label=PRETTY[mode], color=shades[i])
        _label_bars(ax, bars)
    ax.set_xticks(range(len(ks)), [k.replace("hit@", "Hit@") for k in ks])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Share of queries")
    n = metrics[modes[0]]["queries"]
    ax.set_title(f"Corpus-wide retrieval ({n} queries, 28 reports)", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    fig.savefig(OUT / "retrieval_comparison.png")
    plt.close(fig)


def field_accuracy_figure(m: dict) -> None:
    fields = list(m["rag_llm"]["per_field"])[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ys = range(len(fields))
    h = 0.38
    b1 = ax.barh([y + h / 2 for y in ys], [m["rag_llm"]["per_field"][f] for f in fields], h, color=C_MAIN,
                 label=f"RAG + LLM ({m['model'].split('/')[-1]})")
    b2 = ax.barh([y - h / 2 for y in ys], [m["rule_baseline"]["per_field"][f] for f in fields], h, color=C_ALT,
                 label="Rule-based baseline")
    _label_bars(ax, b1, horizontal=True)
    _label_bars(ax, b2, horizontal=True)
    ax.set_yticks(list(ys), [PRETTY[f] for f in fields])
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Exact-match accuracy")
    ax.grid(axis="y", visible=False)
    ax.set_title("Extraction accuracy per field (all 28 reports)", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.4, -0.1))
    fig.tight_layout()
    fig.savefig(OUT / "accuracy_per_field.png")
    plt.close(fig)


def template_figure(m: dict) -> None:
    groups = ["A", "B"]
    names = ["Template A\n(5 companies, seen when\nwriting the rules)", "Template B\n(2 companies, new wording\nand column order)"]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    w = 0.34
    b1 = ax.bar([i - w / 2 for i in range(2)], [m["rule_baseline"]["per_template"][g] for g in groups], w,
                color=C_ALT, label="Rule-based baseline")
    b2 = ax.bar([i + w / 2 for i in range(2)], [m["rag_llm"]["per_template"][g] for g in groups], w,
                color=C_MAIN, label="RAG + LLM")
    _label_bars(ax, b1)
    _label_bars(ax, b2)
    ax.set_xticks(range(2), names, fontsize=8.5)
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Accuracy")
    ax.set_title("Robustness to an unseen report format", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center")
    fig.tight_layout()
    fig.savefig(OUT / "template_robustness.png")
    plt.close(fig)


def triage_figure(m: dict) -> None:
    t = m["triage"]
    parts = [
        ("Auto-accepted, correct", t.get("accepted_correct", 0), C_MAIN),
        ("Auto-accepted, wrong", t.get("accepted_wrong", 0), C_WARN),
        ("Sent to review, correct", t.get("flagged_correct", 0), "#9FB3C8"),
        ("Sent to review, wrong", t.get("flagged_wrong", 0), "#E8B796"),
    ]
    total = sum(p[1] for p in parts)
    fig, ax = plt.subplots(figsize=(7.5, 1.9))
    left = 0
    for label, v, color in parts:
        ax.barh([0], [v], left=left, color=color, label=f"{label} ({v})")
        if v / total > 0.05:
            ax.text(left + v / 2, 0, f"{v / total:.0%}", ha="center", va="center", color="white", fontsize=9)
        left += v
    ax.set_xlim(0, total)
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.set_title(f"Where the {total} extracted values end up", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.3))
    fig.tight_layout()
    fig.savefig(OUT / "review_triage.png", bbox_inches="tight")
    plt.close(fig)


def evidence_figure(extractions: list[dict], doc_id: str) -> None:
    """Render the first page of a report with every extracted value highlighted at its source."""
    ex = next(e for e in extractions if e["doc_id"] == doc_id)
    doc = pymupdf.open(Path("data/reports") / doc_id)
    chunks = {c["chunk_id"]: c for c in json.loads(Path("index/chunks.json").read_text(encoding="utf-8"))}
    for f in ex["fields"].values():
        if not f["raw"] or not f["page"]:
            continue
        page = doc[f["page"] - 1]
        raw = f["raw"].rstrip("km").strip()
        bbox = (chunks.get(f["source_chunk"]) or {}).get("bbox")
        hits = page.search_for(raw, clip=pymupdf.Rect(bbox) + (-10, -2, 10, 2)) if bbox else []
        for rect in (hits or page.search_for(raw))[:1]:
            annot = page.add_rect_annot(rect + (-2, -1, 2, 1))
            annot.set_colors(stroke=(0.76, 0.43, 0.23) if f["needs_review"] else (0.18, 0.36, 0.54))
            annot.set_border(width=1.2)
            annot.update()
    for i in range(min(2, doc.page_count)):
        pix = doc[i].get_pixmap(dpi=110, clip=doc[i].rect + (0, 0, 0, -doc[i].rect.height * (0.18 if i == 0 else 0.45)))
        pix.save(OUT / f"evidence_page{i + 1}.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extractions", default="results/extractions_qwen2.5-3b.json")
    parser.add_argument("--evidence-doc", default="helios_e_mobility_q3_2025.pdf")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    retrieval = Path("results/retrieval_metrics.json")
    if retrieval.exists():
        retrieval_figure(json.loads(retrieval.read_text()))
    extraction = Path("results/extraction_metrics.json")
    if extraction.exists():
        m = json.loads(extraction.read_text())
        field_accuracy_figure(m)
        template_figure(m)
        triage_figure(m)
    if Path(args.extractions).exists():
        evidence_figure(json.loads(Path(args.extractions).read_text(encoding="utf-8")), args.evidence_doc)
    print(f"Figures written to {OUT}/")


if __name__ == "__main__":
    main()
