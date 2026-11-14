"""Structured KPI extraction: retrieve -> prompt -> parse -> normalise -> validate."""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field

from .config import settings
from .ingest import Chunk
from .llm import LLM
from .normalize import normalize_value
from .retrieval import HybridRetriever
from .schema import FIELDS, GROUPS, FieldSpec
from .validation import is_grounded, review_issues

SYSTEM_PROMPT = (
    "You are a financial analyst assistant at a private equity fund. You read excerpts of quarterly reports "
    "sent by portfolio companies and extract key figures. You only use the excerpts provided. "
    "You answer with a single JSON object and nothing else."
)

USER_TEMPLATE = """Company: {company}
Reporting period: {period}

Excerpts:
{excerpts}

Extract the following fields for {period}:
{fields}

Rules:
- Use the figure for {period} only. Ignore prior-year, budget, year-to-date and previous-quarter columns.
- Copy each number exactly as written in the excerpt, including any unit suffix written next to it (k, m, %, x).
- "source" is the id of the excerpt the value comes from, e.g. "E2".
- If a field is not in the excerpts, use null.

Answer with JSON in this format:
{example}"""


@dataclass
class FieldResult:
    name: str
    value: float | int | str | None
    raw: str | None
    source_chunk: str | None
    page: int | None
    grounded: bool
    issues: list[str] = field(default_factory=list)

    @property
    def needs_review(self) -> bool:
        return self.value is None or not self.grounded or bool(self.issues)


@dataclass
class ReportExtraction:
    doc_id: str
    company: str
    period: str
    model: str
    fields: dict[str, FieldResult]
    latency_s: float

    def to_dict(self) -> dict:
        d = asdict(self)
        for name, fr in self.fields.items():
            d["fields"][name]["needs_review"] = fr.needs_review
        return d


def _parse_json(text: str) -> dict:
    """Extract the first JSON object from a model answer (tolerates code fences and trailing text)."""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start == -1:
        return {}
    depth = 0
    for i, ch in enumerate(text[start:], start):
        depth += ch == "{"
        depth -= ch == "}"
        if depth == 0:
            try:
                return json.loads(text[start:i + 1])
            except json.JSONDecodeError:
                return {}
    return {}


class Extractor:
    def __init__(self, retriever: HybridRetriever, llm: LLM, top_k: int | None = None, mode: str = "hybrid_rerank"):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k or settings.top_k_per_field
        self.mode = mode

    def _context_for(self, doc_id: str, specs: list[FieldSpec]) -> list[Chunk]:
        selected: dict[str, Chunk] = {}
        for spec in specs:
            for chunk in self.retriever.search(spec.query, k=self.top_k, mode=self.mode, doc_id=doc_id):
                selected.setdefault(chunk.chunk_id, chunk)
        # Present excerpts in document order: easier to read for the model and for humans.
        return sorted(selected.values(), key=lambda c: int(c.chunk_id.split("::")[1]))

    def extract(self, doc_id: str) -> ReportExtraction:
        start = time.perf_counter()
        doc_chunks = [c for c in self.retriever.chunks if c.doc_id == doc_id]
        company, period = doc_chunks[0].company, doc_chunks[0].period
        results: dict[str, FieldResult] = {}

        for specs in GROUPS.values():
            context = self._context_for(doc_id, specs)
            labels = {f"E{i + 1}": c for i, c in enumerate(context)}
            excerpts = "\n\n".join(f"[{label}] (page {c.page}, {c.kind}, section: {c.section})\n{c.text}"
                                   for label, c in labels.items())
            fields_txt = "\n".join(f"- {s.name}: {s.description}" for s in specs)
            example = json.dumps({s.name: {"value": "...", "source": "E1"} for s in specs})
            prompt = USER_TEMPLATE.format(company=company, period=period, excerpts=excerpts,
                                          fields=fields_txt, example=example)
            answer = _parse_json(self.llm.complete(SYSTEM_PROMPT, prompt, max_tokens=60 + 40 * len(specs)))

            for spec in specs:
                item = answer.get(spec.name) or {}
                if not isinstance(item, dict):
                    item = {"value": item}
                raw = item.get("value")
                raw = None if raw in (None, "", "...") else str(raw)
                source = labels.get(str(item.get("source", "")).strip("[] "))
                # Small models often cite the wrong excerpt id while copying the right number.
                # Resolve the source by checking where the number actually appears; if it appears
                # nowhere in the context, the value stays ungrounded and goes to review.
                if raw is not None and spec.kind != "category" and not (source and is_grounded(raw, source.text)):
                    source = next((c for c in context if is_grounded(raw, c.text)), source)
                chunk_text = source.text if source else ""
                value = normalize_value(spec.kind, raw, chunk_text)
                results[spec.name] = FieldResult(
                    name=spec.name, value=value, raw=raw,
                    source_chunk=source.chunk_id if source else None, page=source.page if source else None,
                    grounded=bool(source) and raw is not None and (spec.kind == "category" or is_grounded(raw, chunk_text)),
                )

        issues = review_issues({k: {"value": r.value, "raw": r.raw} for k, r in results.items()},
                               {s.name: s.kind for s in FIELDS})
        for name, problems in issues.items():
            results[name].issues.extend(problems)
        return ReportExtraction(doc_id, company, period, self.llm.name, results, time.perf_counter() - start)
