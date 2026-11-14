"""Evaluate retrieval and extraction against the ground truth.

    python scripts/evaluate.py retrieval                       # BM25 vs dense vs hybrid vs hybrid+rerank
    python scripts/evaluate.py extraction results/extractions_qwen2.5-3b.json

Outputs go to results/*.json; figures are produced by scripts/make_figures.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finrag.baseline import extract_baseline  # noqa: E402
from finrag.config import settings  # noqa: E402
from finrag.normalize import is_correct  # noqa: E402
from finrag.retrieval import MODES, HybridRetriever  # noqa: E402
from finrag.schema import FIELD_BY_NAME, FIELDS  # noqa: E402
from finrag.validation import is_grounded, review_issues  # noqa: E402

GROUND_TRUTH = json.loads(Path("data/ground_truth.json").read_text(encoding="utf-8"))
RESULTS = Path("results")


def _hit(chunk, doc_id: str, evidence: str) -> bool:
    return chunk.doc_id == doc_id and evidence in chunk.text


def evaluate_retrieval(retriever: HybridRetriever, ks=(1, 3, 5)) -> dict:
    """Corpus-wide retrieval: 336 queries over 280 chunks of very similar documents.

    A query is "<company> <period> <field description>". It is a hit when a retrieved
    chunk belongs to the right report and contains the exact string printed for the value.
    """
    max_k = 10
    metrics = {}
    for mode in MODES:
        hits = {k: 0 for k in ks}
        rr, n = 0.0, 0
        for doc_id, truth in GROUND_TRUTH.items():
            for spec in FIELDS:
                query = f"{truth['company']} {truth['period']} {spec.description}"
                results = retriever.search(query, k=max_k, mode=mode)
                rank = next((i + 1 for i, c in enumerate(results) if _hit(c, doc_id, truth["evidence"][spec.name])), None)
                n += 1
                if rank:
                    rr += 1 / rank
                    for k in ks:
                        hits[k] += rank <= k
        metrics[mode] = {**{f"hit@{k}": hits[k] / n for k in ks}, "mrr@10": rr / n, "queries": n}
        print(f"{mode:<14}" + "  ".join(f"{m}={v:.3f}" for m, v in metrics[mode].items() if m != "queries"))
    return metrics


def _score(predictions: dict[str, dict]) -> dict:
    """predictions: {doc_id: {field: value}} -> accuracy breakdowns."""
    per_field, per_template, per_unit = defaultdict(list), defaultdict(list), defaultdict(list)
    for doc_id, preds in predictions.items():
        truth = GROUND_TRUTH[doc_id]
        for name, spec in FIELD_BY_NAME.items():
            ok = is_correct(spec.kind, preds.get(name), truth["fields"][name])
            per_field[name].append(ok)
            per_template[truth["difficulty"]["template"]].append(ok)
            per_unit[truth["difficulty"]["unit"]].append(ok)
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    all_ok = [x for xs in per_field.values() for x in xs]
    return {
        "accuracy": mean(all_ok), "n": len(all_ok),
        "per_field": {k: mean(v) for k, v in per_field.items()},
        "per_template": {k: mean(v) for k, v in sorted(per_template.items())},
        "per_unit": {k: mean(v) for k, v in sorted(per_unit.items())},
    }


def evaluate_extraction(path: Path, retriever: HybridRetriever) -> dict:
    extractions = json.loads(path.read_text(encoding="utf-8"))
    llm_preds = {e["doc_id"]: {k: f["value"] for k, f in e["fields"].items()} for e in extractions}
    base_preds = {d: extract_baseline([c for c in retriever.chunks if c.doc_id == d]) for d in llm_preds}

    # Review triage: how useful are the "needs review" flags?
    # The checks are deterministic, so they are re-applied with the current code to the stored
    # LLM outputs, so improving a check does not require re-running the model.
    kinds = {s.name: s.kind for s in FIELDS}
    triage = defaultdict(int)
    for e in extractions:
        truth = GROUND_TRUTH[e["doc_id"]]["fields"]
        issues = review_issues(e["fields"], kinds)
        by_id = {c.chunk_id: c for c in retriever.chunks if c.doc_id == e["doc_id"]}
        for name, f in e["fields"].items():
            suffix = "_correct" if is_correct(FIELD_BY_NAME[name].kind, f["value"], truth[name]) else "_wrong"
            if f["raw"] is not None and not f["grounded"] and kinds[name] != "category":
                # re-check grounding with the current matcher: cited excerpt first, then the rest of the report
                cited = by_id.get(f["source_chunk"])
                match = cited if cited and is_grounded(f["raw"], cited.text) else \
                    next((c for c in by_id.values() if is_grounded(f["raw"], c.text)), None)
                if match:
                    f.update(grounded=True, source_chunk=match.chunk_id, page=match.page)
            flagged = f["value"] is None or not f["grounded"] or bool(issues.get(name))
            f["needs_review"], f["issues"] = flagged, issues.get(name, [])
            triage[("flagged" if flagged else "accepted") + suffix] += 1
    # Store the up-to-date review flags next to the (unchanged) model outputs.
    path.write_text(json.dumps(extractions, indent=2, ensure_ascii=False), encoding="utf-8")
    accepted = triage["accepted_correct"] + triage["accepted_wrong"]
    wrong = triage["accepted_wrong"] + triage["flagged_wrong"]

    report = {
        "model": extractions[0]["model"],
        "reports": len(extractions),
        "rag_llm": _score(llm_preds),
        "rule_baseline": _score(base_preds),
        "triage": dict(triage),
        "auto_accept_rate": accepted / sum(triage.values()),
        "accuracy_of_auto_accepted": triage["accepted_correct"] / accepted if accepted else None,
        "errors_caught_by_review": triage["flagged_wrong"] / wrong if wrong else None,
        "mean_latency_s": sum(e["latency_s"] for e in extractions) / len(extractions),
    }
    for name in ("rag_llm", "rule_baseline"):
        s = report[name]
        print(f"{name:<14} accuracy={s['accuracy']:.3f}  template A={s['per_template'].get('A', 0):.3f}  "
              f"template B={s['per_template'].get('B', 0):.3f}")
    print(f"auto-accepted {report['auto_accept_rate']:.1%} of fields, accuracy on those "
          f"{report['accuracy_of_auto_accepted']:.1%}; review flags caught {report['errors_caught_by_review']:.1%} of errors")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=["retrieval", "extraction"])
    parser.add_argument("path", nargs="?", help="extraction results JSON (for 'extraction')")
    args = parser.parse_args()

    retriever = HybridRetriever.load(settings.index_dir)
    RESULTS.mkdir(exist_ok=True)
    if args.what == "retrieval":
        out = evaluate_retrieval(retriever)
        (RESULTS / "retrieval_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    else:
        out = evaluate_extraction(Path(args.path), retriever)
        (RESULTS / "extraction_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
