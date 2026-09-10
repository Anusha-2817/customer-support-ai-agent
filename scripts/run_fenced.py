"""Run src/run_systems.py with the golden labels file fenced off.

    python scripts/run_fenced.py --systems A --local --run-name full-v1-A-live

Every argument is passed straight to run_systems. Opening data/golden/labels_round1.jsonl for
any reason stops the run, and the number of attempts is printed at the end. Generating replies
never needs the labels; this makes that a checked fact rather than a promise.
"""
import builtins
import io
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABELS = (ROOT / "data" / "golden" / "labels_round1.jsonl").resolve()
attempts: list[str] = []
_open = builtins.open


def fenced(file, *args, **kwargs):
    try:
        hit = Path(file).resolve() == LABELS
    except TypeError:
        hit = False
    if hit:
        attempts.append(str(file))
        raise SystemExit("FENCE: generation tried to open the golden labels file")
    return _open(file, *args, **kwargs)


builtins.open = io.open = fenced
sys.argv = [str(ROOT / "src" / "run_systems.py")] + sys.argv[1:]
try:
    runpy.run_path(str(ROOT / "src" / "run_systems.py"), run_name="__main__")
finally:
    print(f"fence: {len(attempts)} attempts to open the golden labels file during generation", flush=True)
