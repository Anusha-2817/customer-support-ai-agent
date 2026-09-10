"""Smoke tests for the provider layer (src/llm.py, src/providers.py).

    python tests/test_llm_modes.py

Guarantees checked: paid providers cannot start outside --live, the openai package
is never imported in local runs, --offline never computes anything, mock output never
reaches the cache, and nothing connects anywhere except localhost. No test loads a
real model or writes to the project cache. Exit code 1 on any failure.
"""
import argparse
import os
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Record every outgoing connection; refuse anything that isn't this machine.
ATTEMPTS: list[str] = []
_real_connect = socket.socket.connect


def _guarded_connect(self, address):
    host = address[0] if isinstance(address, tuple) else str(address)
    ATTEMPTS.append(host)
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ConnectionRefusedError(f"test blocked a connection to {host}")
    return _real_connect(self, address)


socket.socket.connect = _guarded_connect

import llm  # noqa: E402
import providers as P  # noqa: E402

CACHE = ROOT / "cache"
cache_before = sorted(str(p) for p in CACHE.rglob("*")) if CACHE.exists() else []
MSGS = [{"role": "user", "content": "Reply with JSON"}]


def expect_raises(exc, fn):
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__}")


def t_default_is_local():
    llm.set_mode("local")
    assert llm.mode() == "local" and llm.profile() == "local"
    assert llm.spec_for("generator").startswith("ollama:")
    assert llm.spec_for("embedding").startswith("sentence-transformers:")


def t_paid_blocked_at_construction():
    expect_raises(P.PaidCallBlocked, lambda: P.make_llm("openai:gpt-4.1-mini-2025-04-14", allow_paid=False))
    expect_raises(P.PaidCallBlocked, lambda: P.make_embedder("openai:text-embedding-3-small@512", allow_paid=False))


def t_paid_blocked_even_with_live_profile():
    llm.set_mode("local", profile="live")  # OpenAI models configured, but no --live
    ok, why = llm.llm_available("generator")
    assert not ok and "paid" in why, why
    expect_raises(P.PaidCallBlocked, lambda: llm.chat("generator", MSGS))
    expect_raises(P.PaidCallBlocked, lambda: llm.embed(["x"]))
    llm.set_mode("local")


def t_local_llm_status_reported():
    ok, why = llm.llm_available("generator")
    print(f"      local generator {llm.spec_for('generator')}: {'available' if ok else why}")


def t_mock_chat_parsed_and_not_cached():
    llm.use_provider("generator", P.MockProvider(lambda m: '```json\n{"ok": true}\n```'))
    assert llm.chat_json("generator", MSGS) == {"ok": True}
    assert llm.run_info()["models"]["generator"]["spec"] == "mock:v1"
    llm.clear_overrides()


def t_mock_embed_normalised():
    llm.use_provider("embedding", P.MockEmbedder(dims=16))
    M = llm.embed(["lost bag", "lost bag", "great crew"])
    assert M.shape == (3, 16)
    assert abs(float((M[0] ** 2).sum()) - 1) < 1e-3 and float(M[0] @ M[1]) > 0.999
    llm.clear_overrides()


def t_offline_computes_nothing():
    llm.set_mode("offline")
    n = len(ATTEMPTS)
    expect_raises(llm.CacheMiss, lambda: llm.chat("generator", MSGS))
    expect_raises(llm.CacheMiss, lambda: llm.embed(["a text nobody has embedded 7f3a"]))
    ok, why = llm.llm_available("judge")
    assert not ok and "offline" in why, why
    assert len(ATTEMPTS) == n, f"offline mode opened connections: {ATTEMPTS[n:]}"
    llm.set_mode("local")


def t_parse_json_fallbacks():
    assert llm.parse_json('Sure! Here it is: {"a": 1} hope that helps') == {"a": 1}
    expect_raises(ValueError, lambda: llm.parse_json("no json here"))


def t_mode_flags():
    ap = argparse.ArgumentParser()
    llm.add_mode_args(ap)
    for argv, want in [([], os.environ.get("AGENT_MODE", "local")), (["--offline"], "offline"), (["--local"], "local")]:
        llm.apply_mode_args(ap.parse_args(argv))
        assert llm.mode() == want, (argv, llm.mode())
    sys.stderr, saved = open(os.devnull, "w"), sys.stderr
    try:
        expect_raises(SystemExit, lambda: ap.parse_args(["--offline", "--live"]))
    finally:
        sys.stderr = saved
    llm.set_mode("local")


def t_openai_never_imported():
    assert "openai" not in sys.modules, "openai was imported during local/offline work"


def t_only_localhost_contacted():
    remote = [h for h in ATTEMPTS if h not in ("127.0.0.1", "localhost", "::1")]
    assert not remote, f"attempted remote connections: {remote}"


def t_cache_untouched():
    after = sorted(str(p) for p in CACHE.rglob("*")) if CACHE.exists() else []
    assert after == cache_before, "tests wrote to the project cache"


TESTS = [t_default_is_local, t_paid_blocked_at_construction, t_paid_blocked_even_with_live_profile,
         t_local_llm_status_reported, t_mock_chat_parsed_and_not_cached, t_mock_embed_normalised,
         t_offline_computes_nothing, t_parse_json_fallbacks, t_mode_flags,
         t_openai_never_imported, t_only_localhost_contacted, t_cache_untouched]

if __name__ == "__main__":
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed | connections attempted: {sorted(set(ATTEMPTS)) or 'none'}")
    sys.exit(1 if failed else 0)
