"""Model providers behind one small interface.

Nothing outside this file and src/llm.py knows which vendor runs a model. Code asks
src/llm.py for a *role* ("generator", "judge", "namer", "embedding"), and
config/models.json maps each role to a provider spec such as "ollama:qwen2.5:3b"
or "sentence-transformers:all-MiniLM-L6-v2".

LLM providers
- OllamaProvider   free, runs on this machine; needs Ollama installed and running
- OpenAIProvider   paid API; can only be constructed in --live mode
- MockProvider     deterministic canned output, for smoke tests only

Embedders
- SentenceTransformerEmbedder   free, local, CPU
- OpenAIEmbedder                paid API; --live mode only
- MockEmbedder                  deterministic hash vectors, for smoke tests only

The paid providers take allow_paid and raise PaidCallBlocked without it, so a
misconfigured profile can never spend money silently.
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"


class ProviderUnavailable(RuntimeError):
    """The configured model can't be used right now (e.g. Ollama isn't installed)."""


class PaidCallBlocked(RuntimeError):
    """A paid provider was requested outside --live mode."""


def load_dotenv(path: Path = ENV_FILE) -> None:
    """Read KEY=VALUE lines from the project .env into os.environ.

    .env overrides variables already set in the shell, so a stale key set elsewhere
    on the machine can't silently win. Only the paid provider reads it; local and
    offline runs never touch .env."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ[k.strip().removeprefix("export ").strip()] = v.strip().strip('"').strip("'")


# ---------------------------------------------------------------------------- LLMs
class LLMProvider(ABC):
    spec: str             # "vendor:model"; part of every cache key
    max_workers: int = 1  # sensible concurrency for this provider

    def available(self) -> tuple[bool, str]:
        """(usable now?, reason if not)."""
        return True, ""

    @abstractmethod
    def complete(self, messages: list[dict], *, json_mode: bool = False, temperature: float = 0.0,
                 seed: int = 0, max_tokens: int | None = None) -> tuple[str, dict]:
        """Return (text, usage) with usage = {"in": prompt tokens, "out": output tokens}."""


class OllamaProvider(LLMProvider):
    """A local model served by Ollama (https://ollama.com). Free; runs on this machine."""
    max_workers = 1  # CPU-only inference serves one request at a time anyway

    def __init__(self, model: str, spec: str, host: str | None = None, timeout: int = 900):
        self.model, self.spec, self.timeout = model, spec, timeout
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")

    def _request(self, path: str, body: dict | None = None, timeout: float = 5) -> dict:
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(self.host + path, data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def available(self) -> tuple[bool, str]:
        try:
            tags = self._request("/api/tags")
        except (urllib.error.URLError, OSError):
            return False, f"Ollama is not running at {self.host} (is it installed?)"
        names = {m.get("name") for m in tags.get("models", [])}
        if self.model not in names and f"{self.model}:latest" not in names:
            return False, f"model {self.model!r} is not pulled; run: ollama pull {self.model}"
        return True, ""

    def complete(self, messages, *, json_mode=False, temperature=0.0, seed=0, max_tokens=None):
        options = {"temperature": temperature, "seed": seed}
        if max_tokens:
            options["num_predict"] = max_tokens
        body = {"model": self.model, "messages": messages, "stream": False, "options": options}
        if json_mode:
            body["format"] = "json"
        try:
            r = self._request("/api/chat", body, timeout=self.timeout)
        except (urllib.error.URLError, OSError) as e:
            raise ProviderUnavailable(f"{self.spec}: {e}") from e
        return r["message"]["content"], {"in": r.get("prompt_eval_count", 0), "out": r.get("eval_count", 0)}


class OpenAIProvider(LLMProvider):
    """Paid OpenAI API. Kept as an optional provider; constructed only in --live mode."""
    max_workers = 8

    def __init__(self, model: str, spec: str, *, allow_paid: bool):
        if not allow_paid:
            raise PaidCallBlocked(f"{spec} is a paid API and is used only with --live")
        self.model, self.spec, self._client = model, spec, None

    def _c(self):
        if self._client is None:
            load_dotenv()
            from openai import OpenAI  # imported lazily: local/offline runs never load it
            self._client = OpenAI(max_retries=5)
        return self._client

    def complete(self, messages, *, json_mode=False, temperature=0.0, seed=0, max_tokens=None):
        kw = {"model": self.model, "messages": messages, "seed": seed}
        if not self.model.startswith("gpt-5"):  # gpt-5 models accept only the default temperature
            kw["temperature"] = temperature
        if max_tokens:
            kw["max_completion_tokens"] = max_tokens
        if json_mode:
            kw["response_format"] = {"type": "json_object"}
        r = self._c().chat.completions.create(**kw)
        return r.choices[0].message.content, {"in": r.usage.prompt_tokens, "out": r.usage.completion_tokens}


class MockProvider(LLMProvider):
    """Deterministic stand-in for smoke tests: responder(messages) returns the reply
    text. Its outputs are never cached and never reported as results."""
    max_workers = 4

    def __init__(self, responder: Callable[[list[dict]], str], spec: str = "mock:v1"):
        self.responder, self.spec = responder, spec

    def complete(self, messages, *, json_mode=False, temperature=0.0, seed=0, max_tokens=None):
        return self.responder(messages), {"in": 0, "out": 0}


# ------------------------------------------------------------------------ embedders
class Embedder(ABC):
    spec: str

    def available(self) -> tuple[bool, str]:
        return True, ""

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Float32 matrix, one row per text. Callers L2-normalise."""


class SentenceTransformerEmbedder(Embedder):
    """Free local embeddings. all-MiniLM-L6-v2: 384 dims, ~90 MB, Apache-2.0, fast on
    CPU, English-only; inputs are truncated at 256 word pieces, which tweets fit."""

    def __init__(self, model: str, spec: str, revision: str | None = None):
        self.model, self.spec, self.revision, self._m = model, spec, revision, None

    def available(self) -> tuple[bool, str]:
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            return False, "sentence-transformers is not installed (pip install -r requirements-local.txt)"
        except Exception as e:  # noqa: BLE001 - e.g. a broken or half-installed torch
            return False, f"sentence-transformers failed to import: {type(e).__name__}: {e}"
        return True, ""

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._m is None:
            ok, why = self.available()
            if not ok:
                raise ProviderUnavailable(why)
            from sentence_transformers import SentenceTransformer
            self._m = SentenceTransformer(f"sentence-transformers/{self.model}",
                                          revision=self.revision, device="cpu")
        return self._m.encode(texts, batch_size=64, convert_to_numpy=True,
                              show_progress_bar=len(texts) > 2000).astype(np.float32)


class OpenAIEmbedder(Embedder):
    """Paid OpenAI embeddings; --live mode only."""

    def __init__(self, model: str, dims: int, spec: str, *, allow_paid: bool):
        if not allow_paid:
            raise PaidCallBlocked(f"{spec} is a paid API and is used only with --live")
        self.model, self.dims, self.spec, self._client = model, dims, spec, None

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._client is None:
            load_dotenv()
            from openai import OpenAI
            self._client = OpenAI(max_retries=5)
        out = []
        for i in range(0, len(texts), 256):
            r = self._client.embeddings.create(model=self.model, input=texts[i:i + 256], dimensions=self.dims)
            out += [d.embedding for d in r.data]
        return np.asarray(out, dtype=np.float32)


class MockEmbedder(Embedder):
    """Deterministic pseudo-random vectors from a hash of the text. Smoke tests only:
    similar texts do NOT get similar vectors."""

    def __init__(self, dims: int = 32, spec: str = "mock:hash-embed"):
        self.dims, self.spec = dims, spec

    def encode(self, texts: list[str]) -> np.ndarray:
        rows = []
        for t in texts:
            seed = int.from_bytes(hashlib.sha256(t.encode("utf-8")).digest()[:8], "little")
            rows.append(np.random.default_rng(seed).standard_normal(self.dims))
        return np.asarray(rows, dtype=np.float32)


# ------------------------------------------------------------------------- factory
def make_llm(spec: str, *, allow_paid: bool) -> LLMProvider:
    """spec = "vendor:model", e.g. "ollama:qwen2.5:3b" (model names may contain ':')."""
    vendor, _, model = spec.partition(":")
    if vendor == "ollama":
        return OllamaProvider(model, spec)
    if vendor == "openai":
        return OpenAIProvider(model, spec, allow_paid=allow_paid)
    raise ValueError(f"unknown LLM provider in spec {spec!r}")


def make_embedder(spec: str, *, allow_paid: bool) -> Embedder:
    """spec = "vendor:model[@extra]"; extra is a pinned revision (sentence-transformers)
    or the output dimensions (openai)."""
    vendor, _, rest = spec.partition(":")
    model, _, extra = rest.partition("@")
    if vendor == "sentence-transformers":
        return SentenceTransformerEmbedder(model, spec, revision=extra or None)
    if vendor == "openai":
        return OpenAIEmbedder(model, int(extra or 512), spec, allow_paid=allow_paid)
    raise ValueError(f"unknown embedding provider in spec {spec!r}")
