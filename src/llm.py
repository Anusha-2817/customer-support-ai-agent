"""Cached, provider-agnostic model access. Every model call in the project goes
through this module.

Code asks for a ROLE, never a vendor:
    llm.chat_json("generator", messages)        llm.embed(texts)
config/models.json maps each role to a model for the active profile.

Modes (pick one per run with --local / --offline / --live; see add_mode_args):
  local    Default. Free models on this machine: sentence-transformers for embeddings,
           Ollama for LLM roles. Never calls a paid API.
  offline  Replay the cache only. No model runs and no network is used; a cache miss
           is an error. Reproducing results this way needs only numpy, pandas and
           scikit-learn.
  live     Allows the paid OpenAI API ("live" profile). Must be requested explicitly
           and is never the default; paid providers refuse to start otherwise.

Cache (same design as before): chat responses in cache/chat.jsonl, embeddings in
cache/embeddings/<spec>.npz, keyed by a hash of (model spec, input, params). The spec
names the provider, so outputs from different models can never be mixed up. Mock
(test) outputs are never written to the cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import providers as P

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "cache"
CHAT_CACHE = CACHE_DIR / "chat.jsonl"
EMB_DIR = CACHE_DIR / "embeddings"
MODELS = ROOT / "config" / "models.json"
MODES = ("local", "offline", "live")


class CacheMiss(RuntimeError):
    """--offline mode needed a result that is not in the cache."""


_state = {"mode": os.environ.get("AGENT_MODE", "local"), "profile": None}
_lock = threading.Lock()
_chat: dict[str, dict] | None = None
_emb: dict[str, dict[str, np.ndarray]] = {}
_llms: dict[str, P.LLMProvider] = {}
_embedders: dict[str, P.Embedder] = {}
_overrides: dict[str, object] = {}   # role -> injected provider (smoke tests)


# ----------------------------------------------------------------------------- modes
def set_mode(mode: str, profile: str | None = None) -> None:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
    _state.update(mode=mode, profile=profile)
    _llms.clear()
    _embedders.clear()


def mode() -> str:
    return _state["mode"]


def profile() -> str:
    """Which column of config/models.json is in use."""
    return _state["profile"] or ("live" if mode() == "live" else "local")


def add_mode_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--local", dest="mode", action="store_const", const="local",
                   help="free local models (default)")
    g.add_argument("--offline", dest="mode", action="store_const", const="offline",
                   help="replay the cache only: no model runs, no network")
    g.add_argument("--live", dest="mode", action="store_const", const="live",
                   help="allow the PAID OpenAI API for cache misses")
    parser.add_argument("--profile", choices=["local", "live"],
                        help="with --offline: whose cache to replay (default: local)")
    parser.set_defaults(mode=None)


def apply_mode_args(args: argparse.Namespace) -> None:
    set_mode(args.mode or os.environ.get("AGENT_MODE", "local"), getattr(args, "profile", None))
    if mode() == "live":
        print("WARNING: --live mode. Cache misses will make PAID OpenAI API calls.", flush=True)


def spec_for(role: str) -> str:
    if role in _overrides:
        return _overrides[role].spec
    cfg = json.loads(MODELS.read_text(encoding="utf-8"))
    try:
        return cfg[profile()][role]
    except KeyError:
        raise KeyError(f"no model configured for role {role!r} in profile {profile()!r}") from None


def use_provider(role: str, provider) -> None:
    """Inject a provider for a role (smoke tests use MockProvider / MockEmbedder)."""
    _overrides[role] = provider


def clear_overrides() -> None:
    _overrides.clear()


def _key(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------------ chat
def _load_chat() -> dict[str, dict]:
    global _chat
    if _chat is None:
        _chat = {}
        if CHAT_CACHE.exists():
            with open(CHAT_CACHE, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        _chat[rec["key"]] = rec
    return _chat


def _llm(role: str) -> P.LLMProvider:
    if role in _overrides:
        return _overrides[role]
    if role not in _llms:
        _llms[role] = P.make_llm(spec_for(role), allow_paid=(mode() == "live"))
    return _llms[role]


def llm_available(role: str) -> tuple[bool, str]:
    """Can this role produce output right now? In offline mode: only if the cache
    holds responses from its model."""
    if role in _overrides:
        return True, ""
    spec = spec_for(role)
    if mode() == "offline":
        has = any(r.get("spec") == spec for r in _load_chat().values())
        return has, "" if has else f"offline mode and no cached responses from {spec}"
    try:
        return _llm(role).available()
    except P.PaidCallBlocked as e:
        return False, str(e)


def chat(role: str, messages: list[dict], *, json_mode: bool = False, temperature: float = 0.0,
         seed: int = 0, max_tokens: int | None = None) -> str:
    """One completion for a role, served from the cache when possible."""
    spec = spec_for(role)
    params = {"json_mode": json_mode, "temperature": temperature, "seed": seed, "max_tokens": max_tokens}
    k = _key({"spec": spec, "messages": messages, **params})
    cache = _load_chat()
    if k in cache:
        return cache[k]["response"]
    if mode() == "offline" and role not in _overrides:
        raise CacheMiss(f"offline mode: no cached response from {spec}")
    text, usage = _llm(role).complete(messages, **params)
    if not spec.startswith("mock:"):
        rec = {"key": k, "spec": spec, "role": role, "response": text, "usage": usage,
               "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with _lock:
            cache[k] = rec
            CACHE_DIR.mkdir(exist_ok=True)
            with open(CHAT_CACHE, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return text


def parse_json(text: str) -> dict:
    """Parse a JSON object. Local models sometimes wrap it in prose or code fences,
    so fall back to the outermost {...}."""
    try:
        v = json.loads(text)
    except json.JSONDecodeError:
        i, j = text.find("{"), text.rfind("}")
        if i == -1 or j <= i:
            raise ValueError(f"no JSON object in model output: {text[:120]!r}") from None
        v = json.loads(text[i:j + 1])
    if not isinstance(v, dict):
        raise ValueError("model output is JSON but not an object")
    return v


def chat_json(role: str, messages: list[dict], **params) -> dict:
    return parse_json(chat(role, messages, json_mode=True, **params))


def _workers_for(role: str) -> int:
    if role in _overrides:
        return _overrides[role].max_workers
    return 8 if mode() == "offline" else _llm(role).max_workers


def chat_many(requests: list[dict], workers: int | None = None) -> list[dict]:
    """chat_json for each request dict ({"role", "messages", optional params}); results
    in input order. A failed call yields {"_error": ...} instead of killing the batch,
    so one bad response can't silently drop an example."""
    def one(req: dict) -> dict:
        try:
            return chat_json(**req)
        except Exception as e:  # noqa: BLE001 - surfaced to the caller, not swallowed
            return {"_error": f"{type(e).__name__}: {e}"}

    if not requests:
        return []
    workers = workers or min(_workers_for(r["role"]) for r in requests)
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, requests))


def usage_report() -> dict[str, dict]:
    """Calls and tokens per model spec, from the cache."""
    out: dict[str, dict] = {}
    for rec in _load_chat().values():
        m = out.setdefault(rec.get("spec", "?"), {"calls": 0, "in": 0, "out": 0})
        m["calls"] += 1
        m["in"] += rec["usage"].get("in", 0)
        m["out"] += rec["usage"].get("out", 0)
    return out


# ------------------------------------------------------------------------ embeddings
def _emb_path(spec: str) -> Path:
    return EMB_DIR / (re.sub(r"[^A-Za-z0-9._-]+", "_", spec) + ".npz")


def _load_emb(spec: str) -> dict[str, np.ndarray]:
    if spec not in _emb:
        _emb[spec] = {}
        p = _emb_path(spec)
        if p.exists():
            z = np.load(p)
            _emb[spec] = dict(zip(z["keys"].tolist(), z["vecs"]))
    return _emb[spec]


def _save_emb(spec: str) -> None:
    store = _emb[spec]
    keys = list(store)
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(_emb_path(spec), keys=np.array(keys),
                        vecs=np.stack([store[k] for k in keys]).astype(np.float16))


def _embedder() -> P.Embedder:
    if "embedding" in _overrides:
        return _overrides["embedding"]
    spec = spec_for("embedding")
    if spec not in _embedders:
        _embedders[spec] = P.make_embedder(spec, allow_paid=(mode() == "live"))
    return _embedders[spec]


def embed_available() -> tuple[bool, str]:
    if "embedding" in _overrides:
        return True, ""
    spec = spec_for("embedding")
    if mode() == "offline":
        ok = _emb_path(spec).exists()
        return ok, "" if ok else f"offline mode and no cached embeddings from {spec}"
    try:
        return _embedder().available()
    except P.PaidCallBlocked as e:
        return False, str(e)


def embed(texts: list[str], batch: int = 512) -> np.ndarray:
    """Embed texts -> float32 matrix with L2-normalised rows (dot product = cosine)."""
    spec = spec_for("embedding")
    cache = _load_emb(spec)
    texts = [t if t and t.strip() else " " for t in texts]
    keys = [_key({"spec": spec, "t": t}) for t in texts]
    todo = list({k: t for k, t in zip(keys, texts) if k not in cache}.items())
    if todo:
        if mode() == "offline" and "embedding" not in _overrides:
            raise CacheMiss(f"offline mode: {len(todo)} texts have no cached embedding from {spec}")
        enc = _embedder()
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            for (k, _), v in zip(chunk, enc.encode([t for _, t in chunk])):
                cache[k] = v.astype(np.float16)
        if not spec.startswith("mock:"):
            _save_emb(spec)
    M = np.stack([cache[k] for k in keys]).astype(np.float32)
    M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    return M


# -------------------------------------------------------------------------- run info
def run_info(roles: tuple[str, ...] = ("embedding", "generator", "judge", "namer")) -> dict:
    """What produced a result: mode, profile, and each role's model and availability.
    Written next to every result so local and paid runs are never confused."""
    info = {"mode": mode(), "profile": profile(), "models": {}}
    for r in roles:
        try:
            spec = spec_for(r)
        except KeyError:
            continue
        ok, why = embed_available() if r == "embedding" else llm_available(r)
        info["models"][r] = {"spec": spec, "available": ok, **({"why_not": why} if why else {})}
    return info
