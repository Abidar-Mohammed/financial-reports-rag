"""Free-form question answering over the whole report corpus, with citations."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .ingest import Chunk
from .llm import LLM
from .retrieval import HybridRetriever

SYSTEM = (
    "You answer questions from private equity analysts using only the report excerpts provided. "
    "Cite the excerpts you use with their id in square brackets, e.g. [E2]. "
    "If the excerpts do not contain the answer, say so. Be concise and give figures with their unit."
)


@dataclass
class Answer:
    question: str
    text: str
    sources: list[Chunk]

    def format(self) -> str:
        cited = sorted(set(int(n) for n in re.findall(r"\[E(\d+)\]", self.text)))
        lines = [self.text, "", "Sources:"]
        for n in cited or range(1, len(self.sources) + 1):
            if 1 <= n <= len(self.sources):
                c = self.sources[n - 1]
                lines.append(f"  [E{n}] {c.doc_id}, page {c.page}, {c.section}")
        return "\n".join(lines)


def answer_question(question: str, retriever: HybridRetriever, llm: LLM, k: int = 5) -> Answer:
    chunks = retriever.search(question, k=k, mode="hybrid_rerank")
    excerpts = "\n\n".join(f"[E{i + 1}] {c.full_text}" for i, c in enumerate(chunks))
    text = llm.complete(SYSTEM, f"Excerpts:\n{excerpts}\n\nQuestion: {question}", max_tokens=300).strip()
    return Answer(question, text, chunks)
