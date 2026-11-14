"""Download the quantised LLM used by default (about 2 GB) into ./models.

The embedding and reranking models are small and are downloaded automatically
by sentence-transformers on first use.
"""
from pathlib import Path

from huggingface_hub import hf_hub_download

MODELS = {
    "qwen2.5-3b": ("Qwen/Qwen2.5-3B-Instruct-GGUF", "qwen2.5-3b-instruct-q4_k_m.gguf"),
}

if __name__ == "__main__":
    Path("models").mkdir(exist_ok=True)
    for repo, filename in MODELS.values():
        print("downloading", repo, filename)
        print(hf_hub_download(repo, filename, local_dir="models"))
