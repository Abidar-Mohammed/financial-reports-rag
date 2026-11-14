"""Command line interface.

    python -m finrag index   data/reports            # parse PDFs and build the index
    python -m finrag extract data/reports/x.pdf      # extract KPIs from one report
    python -m finrag extract --all --out results/extractions.json
    python -m finrag ask "What was Helios' leverage in Q3 2025?"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import settings


def _retriever():
    from .retrieval import HybridRetriever

    if not (settings.index_dir / "chunks.json").exists():
        raise SystemExit(f"No index found in {settings.index_dir}. Run `python -m finrag index data/reports` first.")
    return HybridRetriever.load(settings.index_dir)


def cmd_index(args) -> None:
    from .ingest import parse_directory
    from .retrieval import HybridRetriever

    chunks = parse_directory(Path(args.reports), settings.max_chunk_words)
    HybridRetriever(chunks).save(settings.index_dir)
    docs = len({c.doc_id for c in chunks})
    tables = sum(c.kind == "table" for c in chunks)
    print(f"Indexed {len(chunks)} chunks ({tables} tables) from {docs} reports into {settings.index_dir}/")


def _print_extraction(ex) -> None:
    print(f"\n{ex.company} - {ex.period}   ({ex.model}, {ex.latency_s:.1f}s)")
    print(f"{'field':<20}{'value':>14}  {'page':>4}  status")
    for f in ex.fields.values():
        value = "-" if f.value is None else (f"{f.value:,.3f}" if isinstance(f.value, float) else str(f.value))
        status = "REVIEW " + "; ".join(f.issues or (["not grounded"] if not f.grounded else ["missing"])) \
            if f.needs_review else "ok"
        print(f"{f.name:<20}{value:>14}  {f.page or '-':>4}  {status}")


def cmd_extract(args) -> None:
    from .extraction import Extractor
    from .llm import get_llm

    retriever = _retriever()
    extractor = Extractor(retriever, get_llm(args.provider, args.model))
    doc_ids = sorted({c.doc_id for c in retriever.chunks}) if args.all else [Path(p).name for p in args.reports]
    out = Path(args.out) if args.out else None
    results = []
    if out and out.exists():  # resume: keep reports already extracted
        results = json.loads(out.read_text(encoding="utf-8"))
        done = {r["doc_id"] for r in results}
        if done:
            print(f"Resuming: {len(done)} reports already in {out}")
        doc_ids = [d for d in doc_ids if d not in done]
    total = len(results) + len(doc_ids)
    for doc_id in doc_ids:
        print(f"\n[{len(results) + 1}/{total}] {doc_id}", flush=True)
        ex = extractor.extract(doc_id)
        _print_extraction(ex)
        results.append(ex.to_dict())
        if out:  # save after every report so an interruption loses nothing
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    if out:
        print(f"\nSaved {len(results)} extractions to {out}")


def cmd_ask(args) -> None:
    from .llm import get_llm
    from .qa import answer_question

    print(answer_question(args.question, _retriever(), get_llm(args.provider, args.model)).format())


def main() -> None:
    parser = argparse.ArgumentParser(prog="finrag", description="RAG pipeline for portfolio company reports")
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("index", help="parse PDFs and build the retrieval index")
    p.add_argument("reports", help="directory containing PDF reports")
    p.set_defaults(func=cmd_index)

    for name, func in (("extract", cmd_extract), ("ask", cmd_ask)):
        p = sub.add_parser(name)
        p.add_argument("--provider", choices=["llamacpp", "groq", "ollama", "local"], default=None)
        p.add_argument("--model", default=None)
        p.set_defaults(func=func)
        if name == "extract":
            p.add_argument("reports", nargs="*", help="PDF files (already indexed)")
            p.add_argument("--all", action="store_true", help="extract every indexed report")
            p.add_argument("--out", help="write results to a JSON file")
        else:
            p.add_argument("question")

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
