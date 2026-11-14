# Design notes

Why the pipeline is built the way it is, what was tried, and what I would change for a production deployment.

## Problem framing

In a private equity fund, every portfolio company sends a monthly or quarterly report. Analysts copy
the same dozen KPIs out of each one into the fund's monitoring model: revenue, EBITDA, net debt,
leverage, covenant status, guidance. The reports are PDFs, the layout differs from one company to
the next, and the same metric goes by several names. The job is repetitive, but mistakes are costly
because the numbers end up in investment committee material and in LP reporting.

So the goal is not a chatbot. It is a tool that:

1. pre-fills the KPI table for each report,
2. shows where each number comes from (page and excerpt),
3. tells the analyst which values to double-check.

Point 3 matters most. A system that is right 95% of the time but cannot say *which* 5% is wrong still
forces a full manual check.

## Decisions

### Layout-aware chunking instead of fixed windows
A fixed 500-character window cuts tables in half and separates a number from its column header,
so the model cannot tell the actual figure from the prior-year one. Tables are detected with
PyMuPDF and kept as single chunks. The caption above each table goes with it, because that is
where the unit lives (`EUR '000`). Narrative text is grouped by section heading.

Every chunk is prefixed with `company | period | section`. Without that header, a chunk saying
"Net debt | 59,682" is impossible to attribute once it is retrieved from a corpus of 28 near-identical
reports.

### Hybrid retrieval with a reranker
- BM25 handles exact tokens: company names, "Q3 2025", "EBITDA".
- Dense embeddings (bge-small) handle synonyms: "turnover", "net sales" and "revenue".
- Reciprocal Rank Fusion merges both lists without calibrating their scores against each other.
- A cross-encoder reranks the top 20 candidates.

`results/retrieval_metrics.json` has the measured effect of each component.

### Grouped extraction calls
One call per field costs 12 LLM calls per report. One call for everything gives a long prompt that
small models handle badly. Fields are grouped by where they usually appear (P&L, balance sheet,
operations, outlook): that is 4 calls per report with short, focused contexts.

### The LLM copies, the code converts
The model returns numbers *exactly as written*, plus the id of the excerpt it used. Unit
conversion (thousands to millions) is done in code from the cited excerpt. Asking a 3B model to
multiply by 1,000 is an unnecessary source of errors. It also makes the next check possible.

### Grounding check
If the number returned by the model does not appear literally in the context, it was invented or
recomputed. That field is marked "not grounded" and sent to review. Small models often cite the
wrong excerpt id while copying the right number, so the source is re-resolved by looking for the
number in the other excerpts before the field is flagged.

### Accounting consistency checks
Values have to agree with each other: margin = EBITDA / revenue, guidance low <= high, and
full-year guidance in line with quarterly revenue. This catches unit errors and column mix-ups
(prior year instead of actual) that grounding alone cannot catch, since the wrong number *is*
in the document.

### Free, local-first models
Everything runs on a laptop CPU with no API key: a quantised Qwen2.5-3B served through llama.cpp,
plus bge-small for embeddings. The LLM layer is a small interface, so switching to Groq (free tier,
Llama 3.3 70B) or a local Ollama server only takes a setting in `.env`. In a real fund, report
confidentiality usually makes a self-hosted model the only acceptable option anyway.

## Evaluation methodology

Real portfolio reports are confidential, so the corpus is synthetic (`scripts/generate_reports.py`).
The upside is exact ground truth. The generator deliberately includes the difficulties seen in real
reports: synonyms, thousands vs millions, prior-year and budget columns next to the actuals, KPIs
that only appear in prose, and three currencies.

Two report templates are used. The rule-based baseline was written from template A. Template B
(two other companies) uses different wording and puts the prior-year column first. This measures
something important in practice: what happens when a new portfolio company joins with its own
format.

## Limitations

- Synthetic reports are cleaner than real ones: there are no scanned pages, no multi-page tables
  and no footnotes changing a definition.
- Values are compared with a 0.5% tolerance for amounts and an exact match for percentages and ratios.
- Latency is measured on a 4-core laptop CPU. A GPU or a hosted model is 10 to 50 times faster.

## What I would do next

- OCR fallback (e.g. docTR) for scanned reports.
- Per-company memory: store the label mapping confirmed by the analyst
  ("Recurring EBITDA" = EBITDA for this company) and inject it into the prompt.
- Time-series checks: flag a value that moves more than X% versus the previous quarter of the same company.
- Write the validated values straight into the monitoring model (Excel or database), with an audit trail.
