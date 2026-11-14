"""Thin wrappers around the supported LLM back-ends.

All options are free:
* ``groq``     - hosted open-weight models (Llama 3.3 70B) on Groq's free API tier
* ``ollama``   - any model served by a local Ollama instance
* ``llamacpp`` - a quantised GGUF model run in-process (default; fast enough on a laptop CPU)
* ``local``    - a small instruct model run with transformers (slow without a GPU)
"""
from __future__ import annotations

import time
from typing import Protocol

import requests

from .config import settings


class LLM(Protocol):
    name: str

    def complete(self, system: str, user: str, max_tokens: int = 400) -> str: ...


class GroqLLM:
    url = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, model: str, api_key: str):
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set. Create a free key at console.groq.com and add it to .env")
        self.name, self.api_key = f"groq/{model}", api_key
        self.model = model

    def complete(self, system: str, user: str, max_tokens: int = 400) -> str:
        payload = {"model": self.model, "temperature": 0, "max_tokens": max_tokens,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        for attempt in range(5):
            r = requests.post(self.url, json=payload, timeout=60, headers={"Authorization": f"Bearer {self.api_key}"})
            if r.status_code == 429:  # free tier rate limit: back off and retry
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        raise RuntimeError("Groq rate limit not cleared after retries")


class OllamaLLM:
    def __init__(self, model: str, url: str):
        self.name, self.model, self.url = f"ollama/{model}", model, url.rstrip("/")

    def complete(self, system: str, user: str, max_tokens: int = 400) -> str:
        r = requests.post(f"{self.url}/api/chat", timeout=300, json={
            "model": self.model, "stream": False, "options": {"temperature": 0, "num_predict": max_tokens},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        r.raise_for_status()
        return r.json()["message"]["content"]


class LlamaCppLLM:
    """Quantised GGUF model run in-process with llama.cpp. The practical choice on a CPU-only laptop."""

    def __init__(self, model_path: str, n_ctx: int = 4096):
        import os

        from llama_cpp import Llama

        self.name = f"llamacpp/{os.path.basename(model_path).removesuffix('.gguf')}"
        # Physical cores usually give the best throughput; hyper-threads mostly add contention.
        threads = max(1, (os.cpu_count() or 2) // 2)
        self.model = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=threads, verbose=False)

    def complete(self, system: str, user: str, max_tokens: int = 400) -> str:
        out = self.model.create_chat_completion(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0, max_tokens=max_tokens)
        return out["choices"][0]["message"]["content"]


class LocalLLM:
    """Hugging Face transformers on CPU. Works everywhere, but slow without a GPU."""

    def __init__(self, model: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = f"local/{model.rstrip('/').split('/')[-1]}"
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16)
        self.model.eval()
        self._torch = torch

    def complete(self, system: str, user: str, max_tokens: int = 400) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with self._torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        return self.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def get_llm(provider: str | None = None, model: str | None = None) -> LLM:
    provider = provider or settings.llm_provider
    if provider != settings.llm_provider:
        settings.llm_provider = provider
    model = model or settings.default_model()
    if provider == "groq":
        return GroqLLM(model, settings.groq_api_key)
    if provider == "ollama":
        return OllamaLLM(model, settings.ollama_url)
    if provider == "llamacpp":
        return LlamaCppLLM(model)
    if provider == "local":
        return LocalLLM(model)
    raise ValueError(f"Unknown provider: {provider}")
