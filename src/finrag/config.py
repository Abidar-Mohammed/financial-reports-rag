"""Runtime configuration, read from environment variables (or a local .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env loader so the project has no hard dependency on python-dotenv."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


@dataclass
class Settings:
    # Embedding and reranking models (both run locally on CPU)
    embed_model: str = field(default_factory=lambda: os.getenv("FINRAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    rerank_model: str = field(default_factory=lambda: os.getenv("FINRAG_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))

    # LLM provider: "llamacpp" (GGUF on CPU), "groq" (free API tier), "ollama" (local server)
    # or "local" (transformers)
    llm_provider: str = field(default_factory=lambda: os.getenv("FINRAG_LLM_PROVIDER", "llamacpp"))
    llm_model: str = field(default_factory=lambda: os.getenv("FINRAG_LLM_MODEL", ""))
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    ollama_url: str = field(default_factory=lambda: os.getenv("OLLAMA_URL", "http://localhost:11434"))

    # Retrieval
    top_k_per_field: int = int(os.getenv("FINRAG_TOP_K", "3"))
    max_chunk_words: int = int(os.getenv("FINRAG_MAX_CHUNK_WORDS", "160"))

    index_dir: Path = field(default_factory=lambda: Path(os.getenv("FINRAG_INDEX_DIR", "index")))

    def default_model(self) -> str:
        if self.llm_model:
            return self.llm_model
        return {
            "llamacpp": "models/qwen2.5-3b-instruct-q4_k_m.gguf",
            "groq": "llama-3.3-70b-versatile",
            "ollama": "qwen2.5:7b-instruct",
            "local": "Qwen/Qwen2.5-1.5B-Instruct",
        }[self.llm_provider]


settings = Settings()
