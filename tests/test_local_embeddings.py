"""Local embedding test: the real all-MiniLM-L6-v2 model, then an offline replay.

    python tests/test_local_embeddings.py

1. --local: load the pinned model (downloaded once from HuggingFace, ~91 MB, then kept
   in the huggingface_hub cache), embed three texts, check the vectors make sense and
   land in a temporary embedding cache.
2. --offline, in a fresh process where sentence-transformers cannot be imported and
   every network connection is refused: the same texts must come back from that cache.

The project cache is never touched. Exit code 1 on any failure.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import llm  # noqa: E402

TEXTS = ["my bag never arrived at Heathrow",
         "BA lost my luggage on the flight to Rome",
         "the cabin crew were lovely, thank you"]

REPLAY = r"""
import socket, sys
from pathlib import Path
sys.modules["sentence_transformers"] = None      # any import of it now fails
def refuse(self, address):
    raise ConnectionRefusedError(f"offline test refused a connection to {address}")
socket.socket.connect = refuse
sys.path.insert(0, sys.argv[1] + "/src")
import llm
llm.EMB_DIR = Path(sys.argv[2])
llm.set_mode("offline")
M = llm.embed(sys.argv[3:])
print(M.shape[0], M.shape[1], round(float(M[0] @ M[1]), 4))
"""

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        failures.append(msg)


project_cache = ROOT / "cache" / "embeddings"
before = sorted(p.name for p in project_cache.glob("*")) if project_cache.exists() else []
tmp = Path(tempfile.mkdtemp(prefix="emb_smoke_"))
try:
    llm.EMB_DIR = tmp
    llm.set_mode("local")
    ok, why = llm.embed_available()
    check(ok, f"local embedder available ({llm.spec_for('embedding')}) {why}")

    t0 = time.time()
    M = llm.embed(TEXTS)
    print(f"      embedded {len(TEXTS)} texts in {time.time() - t0:.1f}s (includes model load / first download)")
    check(M.shape == (3, 384), f"shape is (3, 384): {M.shape}")
    check(all(abs(float(v @ v) - 1) < 1e-3 for v in M), "rows are unit length")
    bag_luggage, bag_crew = float(M[0] @ M[1]), float(M[0] @ M[2])
    check(bag_luggage > bag_crew + 0.1,
          f"lost bag is closer to lost luggage ({bag_luggage:.3f}) than to crew praise ({bag_crew:.3f})")
    files = list(tmp.glob("*.npz"))
    check(len(files) == 1 and "all-MiniLM-L6-v2" in files[0].name, f"cache file written: {[f.name for f in files]}")

    r = subprocess.run([sys.executable, "-c", REPLAY, str(ROOT), str(tmp), *TEXTS],
                       capture_output=True, text=True, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    out = r.stdout.split()
    check(r.returncode == 0 and out[:2] == ["3", "384"],
          "offline replay works with no model library and no network" + ("" if r.returncode == 0 else f": {r.stderr.strip()[-200:]}"))
    if r.returncode == 0:
        check(abs(float(out[2]) - bag_luggage) < 0.01, f"replayed vectors match (similarity {out[2]} vs {bag_luggage:.4f})")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

after = sorted(p.name for p in project_cache.glob("*")) if project_cache.exists() else []
check(after == before, "project cache untouched")
print(f"\n{'all passed' if not failures else f'{len(failures)} FAILED'}")
sys.exit(1 if failures else 0)
