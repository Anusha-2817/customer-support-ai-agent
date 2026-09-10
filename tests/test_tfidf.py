"""TF-IDF intent model tests (src/tfidf_intent.py).

    python tests/test_tfidf.py

Trains on the pool's silver labels, checks predictions on clear-cut messages, and reports
cross-validated agreement with the silver labels. That agreement is a sanity check only:
silver labels are not ground truth. Loads no embedding model, writes nothing.
Exit code 1 on any failure.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import tfidf_intent  # noqa: E402

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        failures.append(msg)


clf = tfidf_intent.train_on_silver()
check(len(clf.classes) == 10, f"learns all 10 intents: {len(clf.classes)}")
for text, want in [("my bag never arrived at heathrow", "baggage"),
                   ("can't log in to the app, error every time", "website_app"),
                   ("my avios points are missing from my executive club account", "loyalty_avios"),
                   ("why is BA123 delayed by 3 hours", "flight_disruption")]:
    got, p = clf.predict([text])[0]
    check(got == want, f"{text!r} -> {got} ({p:.2f})")
check(all(0 <= p <= 1 for _, p in clf.predict(["hello", "thanks"])), "confidences are probabilities")
cv = tfidf_intent.silver_agreement_cv()
check(cv["agreement_with_silver"] > cv["majority_share"] + 0.2,
      f"5-fold agreement with silver labels {cv['agreement_with_silver']:.0%} vs majority class {cv['majority_share']:.0%} "
      "(sanity check only)")

print(f"\n{'all passed' if not failures else f'{len(failures)} FAILED'}")
sys.exit(1 if failures else 0)
