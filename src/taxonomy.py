"""Intent taxonomy induction (design C2). Produces a PROPOSAL for a human to curate;
nothing is final until `finalize` writes data/golden/taxonomy.json.

    python src/taxonomy.py propose [--local | --offline] [--naming auto|llm|manual] [--k K]
    python src/taxonomy.py finalize                       # after filling in curation.json

propose
1. Sample 5,000 customer messages from the retrieval pool, never the eval set: the
   taxonomy must not be shaped by the data it will be evaluated on.
2. Embed them with the configured embedding model (free and local by default; cached).
3. k-means for k = 8..20 with the silhouette score for each k. Vectors are
   L2-normalised, so Euclidean k-means groups messages by cosine similarity.
4. Pick the best-silhouette k in 12..20: more clusters than the 8-10 intents we want,
   because merging clusters by hand is easy and splitting them is not.
5. Describe each cluster with distinctive words (TF-IDF, no model involved), the 10
   messages nearest its centre, and 10 random members. Central messages alone make
   every cluster look cleaner than it is.
6. Naming. If an LLM is available for the "namer" role, it suggests a name per cluster
   and an 8-10 intent draft, marked as suggestions and attributed to the model. If not
   (or with --naming manual), no names are produced at all and the report leaves them
   blank for the human. Nothing is invented.
7. Write a curation template: the human defines the intents and maps every cluster.

finalize
Validates artifacts/taxonomy/curation.json and writes data/golden/taxonomy.json (read
by the labelling tool and the agent) plus the cluster -> intent mapping, which later
turns nearest-centroid assignments into silver labels without needing an LLM.

Outputs in artifacts/taxonomy/: sweep.csv, clusters.json, centroids.npy,
sample_assignments.jsonl, proposal.md, curation_template.json, and after finalize
cluster_to_intent.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
POOL = ROOT / "data" / "processed" / "pool.jsonl"
OUT = ROOT / "artifacts" / "taxonomy"
GOLD_TAXONOMY = ROOT / "data" / "golden" / "taxonomy.json"
GOLD_LABELS = ROOT / "data" / "golden" / "labels_round1.jsonl"
N_SAMPLE = 5000
SALT = "ba-taxonomy-2026-09-10"
TOOL_KEY_LIMIT = 12   # the labelling tool has 12 intent shortcuts (1-0, -, =)

NAME_SYSTEM = (
    "You are helping design an intent taxonomy for British Airways customer support on "
    "Twitter. A clustering algorithm grouped the messages below. Describe the group by what "
    "the customers want British Airways to do, not by topic alone. Reply in JSON.")

RULES = """Rules for the taxonomy:
- 8 to 10 intents in total.
- Each intent is defined by what the customer wants British Airways to do.
- Intents describe the request, never urgency or risk. Whether a message needs a human is
  labelled separately, so do not create intents such as "urgent issue" or "escalation".
- Include exactly one catch-all intent, "other / non-actionable", for praise, banter,
  venting with no request, and off-topic messages.
- Every cluster maps to exactly one intent. A mixed cluster goes to its dominant intent.
- A human reading one tweet must be able to tell intents apart: give boundary notes for
  pairs that are easy to confuse."""


# --------------------------------------------------------------------------- propose
def sample_pool(n: int) -> pd.DataFrame:
    """Hash-based pick (same reasoning as the golden sampler, decision log #15)."""
    pool = pd.read_json(POOL, lines=True, convert_dates=False)
    key = pool["case_id"].astype(str).map(
        lambda c: hashlib.sha256(f"{SALT}:{c}".encode()).hexdigest())
    return pool.assign(_h=key).sort_values("_h").head(n).drop(columns="_h").reset_index(drop=True)


def sweep(X: np.ndarray, ks: range) -> tuple[pd.DataFrame, dict[int, KMeans]]:
    rows, models = [], {}
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        sil = float(silhouette_score(X, km.labels_, sample_size=3000, random_state=0))
        rows.append({"k": k, "silhouette": round(sil, 4), "inertia": round(float(km.inertia_), 1)})
        models[k] = km
        print(f"  k={k:2d}  silhouette={sil:.4f}", flush=True)
    return pd.DataFrame(rows), models


def distinctive_terms(texts: list[str], labels: np.ndarray, k: int, n: int = 8) -> dict[int, list[str]]:
    """Words over-represented in each cluster relative to the whole sample."""
    stop = list(ENGLISH_STOP_WORDS | {"link", "amp"})   # <link> is our URL placeholder
    vec = TfidfVectorizer(stop_words=stop, min_df=5, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    T = vec.fit_transform(texts)
    vocab = np.array(vec.get_feature_names_out())
    overall = np.asarray(T.mean(axis=0)).ravel()
    out = {}
    for c in range(k):
        lift = np.asarray(T[labels == c].mean(axis=0)).ravel() - overall
        out[c] = vocab[np.argsort(lift)[::-1][:n]].tolist()
    return out


def describe(df: pd.DataFrame, X: np.ndarray, km: KMeans, terms: dict) -> list[dict]:
    rng = np.random.default_rng(0)
    out = []
    for c in range(km.n_clusters):
        idx = np.where(km.labels_ == c)[0]
        centre = km.cluster_centers_[c] / np.linalg.norm(km.cluster_centers_[c])
        central = idx[np.argsort(-(X[idx] @ centre))[:10]]
        rest = np.setdiff1d(idx, central)
        rand = rng.choice(rest, size=min(10, len(rest)), replace=False)
        out.append({"cluster": c, "size": int(len(idx)), "share": round(len(idx) / len(df), 4),
                    "terms": terms[c],
                    "central": df.customer_msg.iloc[central].tolist(),
                    "random": df.customer_msg.iloc[rand].tolist()})
    return out


def name_clusters(clusters: list[dict]) -> None:
    reqs = []
    for c in clusters:
        msgs = "\n".join(f"- {m}" for m in c["central"] + c["random"])
        user = (f"Distinctive words: {', '.join(c['terms'])}\n\nMessages:\n{msgs}\n\n"
                'Return JSON: {"name": "2-5 word name", '
                '"definition": "one sentence starting The customer wants", '
                '"coherent": true if most messages share one intent else false, '
                '"mixed_with": ["other intents visible in the group"]}')
        reqs.append({"role": "namer", "temperature": 0,
                     "messages": [{"role": "system", "content": NAME_SYSTEM},
                                  {"role": "user", "content": user}]})
    for c, r in zip(clusters, llm.chat_many(reqs)):
        c["llm"] = r


def consolidate(clusters: list[dict]) -> dict:
    lines = []
    for c in clusters:
        r = c.get("llm", {})
        eg = " | ".join(c["central"][:3] + c["random"][:3])
        lines.append(f"Cluster {c['cluster']} ({c['share']:.1%}): {r.get('name')} - {r.get('definition')} "
                     f"[coherent={r.get('coherent')}, mixed_with={r.get('mixed_with')}] e.g. {eg}")
    user = (RULES + "\n\nClusters:\n" + "\n".join(lines) +
            '\n\nReturn JSON: {"intents": [{"id": "snake_case", "name": "...", '
            '"definition": "The customer wants ...", "clusters": [cluster numbers], '
            '"boundary_notes": "how to tell it apart from similar intents"}], '
            '"notes": "anything the curator should know"}')
    return llm.chat_json("namer", [{"role": "system", "content": NAME_SYSTEM},
                                   {"role": "user", "content": user}], temperature=0)


def check_mapping(proposal: dict, k: int) -> list[str]:
    """Report (never silently fix) clusters mapped to zero or several intents."""
    seen: dict[int, list[str]] = {c: [] for c in range(k)}
    for it in proposal.get("intents", []):
        for c in it.get("clusters", []):
            seen.setdefault(int(c), []).append(it.get("id"))
    return [f"cluster {c}: mapped to {v or 'nothing'}" for c, v in seen.items() if len(v) != 1]


def write_proposal(path: Path, sweep_df: pd.DataFrame, k: int, clusters: list[dict], proposal: dict,
                   problems: list[str], info: dict, naming: str) -> None:
    by_c = {c["cluster"]: c for c in clusters}
    models = info["models"]
    namer = models.get("namer", {})
    L = ["# Intent taxonomy proposal (for curation)", "",
         f"Generated by `src/taxonomy.py propose` from {N_SAMPLE:,} retrieval-pool messages, k = {k}.", "",
         f"- Embeddings: `{models['embedding']['spec']}` (mode: {info['mode']})"]
    if naming == "llm":
        L.append(f"- Cluster names and the intent draft are **suggestions by `{namer.get('spec')}`**, "
                 "a small local model. Check every one against the examples; the curated result is yours.")
    else:
        L.append(f"- **No model named anything.** {namer.get('why_not') or 'Manual naming was requested.'} "
                 "Cluster names are blank for you to fill in from the examples.")
    L += ["", "## What to do", "",
          "1. Read each cluster below: distinctive words, messages nearest the centre, random members.",
          "2. Define 8-10 intents, each a distinct request a labeller can spot in one tweet, "
          "including one catch-all \"other / non-actionable\".",
          "3. Copy `curation_template.json` to `curation.json`, fill in the intents, and map every cluster to one.",
          "4. Run `python src/taxonomy.py finalize`. It checks the file and writes `data/golden/taxonomy.json`.", "",
          "Watch for: clusters that mix several requests (labels will disagree there), and a catch-all "
          "that grows large (a real intent may be hiding in it).", "",
          "## k sweep (silhouette: higher = better separated)", "", "| k | silhouette |", "|---|---|"]
    L += [f"| {r.k}{' (chosen)' if r.k == k else ''} | {r.silhouette:.4f} |" for r in sweep_df.itertuples()]
    if proposal:
        L += ["", f"## Suggested intents (by `{namer.get('spec')}`, not final)", "",
              "| # | Intent | Share | Clusters | Definition |", "|---|---|---|---|---|"]
        for i, it in enumerate(proposal.get("intents", []), 1):
            cl = [int(c) for c in it.get("clusters", [])]
            share = sum(by_c[c]["share"] for c in cl if c in by_c)
            L.append(f"| {i} | **{it.get('name')}** (`{it.get('id')}`) | {share:.1%} | "
                     f"{', '.join(map(str, cl))} | {it.get('definition')} |")
        L += ["", "### Suggested boundary notes", ""]
        L += [f"- **{it.get('name')}**: {it.get('boundary_notes')}" for it in proposal.get("intents", [])]
        if problems:
            L += ["", "**Mapping problems in the suggestion:** " + "; ".join(problems)]
    L += ["", "## Clusters", ""]
    for c in sorted(clusters, key=lambda c: -c["size"]):
        r = c.get("llm", {})
        title = f"suggested name: {r.get('name')}" if r.get("name") else "name: ______"
        L += [f"### Cluster {c['cluster']} ({c['size']} msgs, {c['share']:.1%}) - {title}", "",
              f"Distinctive words: {', '.join(c['terms'])}  "]
        if r.get("definition"):
            L.append(f"Suggested definition: {r['definition']}  ")
        if r.get("mixed_with"):
            L.append(f"Suggested as also containing: {', '.join(map(str, r['mixed_with']))}  ")
        L += ["", "Nearest the centre:"] + [f"- {m}" for m in c["central"][:8]]
        L += ["", "Random members:"] + [f"- {m}" for m in c["random"][:8]] + [""]
    path.write_text("\n".join(L), encoding="utf-8")


def curation_template(clusters: list[dict], proposal: dict, info: dict) -> dict:
    suggested = {int(c): it.get("id", "") for it in proposal.get("intents", []) for c in it.get("clusters", [])}
    if proposal:
        intents = [{"id": it.get("id", ""), "name": it.get("name", ""), "definition": it.get("definition", ""),
                    "boundary_notes": it.get("boundary_notes", "")} for it in proposal.get("intents", [])]
    else:
        intents = [{"id": "", "name": "", "definition": "", "boundary_notes": ""} for _ in range(10)]
    return {
        "_how_to": ("Copy this file to curation.json and edit it. Define 8-10 intents (delete unused "
                    "blank ones); ids are snake_case, e.g. flight_disruption. Set every cluster's "
                    "'intent' to one of your ids. Then run: python src/taxonomy.py finalize"),
        "_suggestions_by": info["models"].get("namer", {}).get("spec") if proposal else None,
        "intents": intents,
        "clusters": {str(c["cluster"]): {"intent": suggested.get(c["cluster"], ""), "size": c["size"],
                                         "words": c["terms"][:5], "example": c["central"][0]}
                     for c in sorted(clusters, key=lambda c: c["cluster"])},
    }


def propose(args: argparse.Namespace) -> None:
    llm.apply_mode_args(args)
    ok, why = llm.embed_available()
    if not ok:
        sys.exit(f"cannot embed: {why}")
    OUT.mkdir(parents=True, exist_ok=True)

    df = sample_pool(N_SAMPLE)
    print(f"sampled {len(df):,} pool messages; embedding with {llm.spec_for('embedding')} ...", flush=True)
    t0 = time.time()
    X = llm.embed(df.customer_msg.tolist())
    print(f"  embedded {X.shape} in {time.time() - t0:.0f}s", flush=True)

    print("k-means sweep:", flush=True)
    sweep_df, models = sweep(X, range(8, 21))
    window = sweep_df[sweep_df.k.between(12, 20)]
    k = args.k or int(window.loc[window.silhouette.idxmax(), "k"])
    km = models[k]
    print(f"chosen k = {k}", flush=True)
    clusters = describe(df, X, km, distinctive_terms(df.customer_msg.tolist(), km.labels_, k))

    if args.naming == "manual":
        use_llm = False
    else:
        avail, why = llm.llm_available("namer")
        if args.naming == "llm" and not avail:
            sys.exit(f"--naming llm was requested but the namer model is unavailable: {why}")
        use_llm = avail
    proposal, problems = {}, []
    if use_llm:
        print(f"naming clusters with {llm.spec_for('namer')} ...", flush=True)
        name_clusters(clusters)
        proposal = consolidate(clusters)
        problems = check_mapping(proposal, k)
    else:
        print("no LLM naming: the report leaves cluster names blank for manual curation", flush=True)

    info = llm.run_info(("embedding", "namer"))
    naming = "llm" if use_llm else "manual"
    sweep_df.to_csv(OUT / "sweep.csv", index=False)
    C = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
    np.save(OUT / "centroids.npy", C.astype(np.float32))
    pd.DataFrame({"case_id": df.case_id, "cluster": km.labels_}).to_json(
        OUT / "sample_assignments.jsonl", orient="records", lines=True)
    (OUT / "clusters.json").write_text(json.dumps(
        {"k": k, "n": len(df), "naming": naming, "run_info": info, "clusters": clusters,
         "proposal": proposal, "problems": problems}, indent=2, ensure_ascii=False), encoding="utf-8")
    write_proposal(OUT / "proposal.md", sweep_df, k, clusters, proposal, problems, info, naming)
    (OUT / "curation_template.json").write_text(json.dumps(
        curation_template(clusters, proposal, info), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT / 'proposal.md'} and curation_template.json | naming: {naming}", flush=True)
    if (OUT / "curation.json").exists():
        print("note: an existing curation.json was left untouched", flush=True)


# -------------------------------------------------------------------------- finalize
def finalize(args: argparse.Namespace) -> None:
    cur_path = OUT / "curation.json"
    if not cur_path.exists():
        sys.exit("artifacts/taxonomy/curation.json not found: copy curation_template.json to it and fill it in")
    if GOLD_LABELS.exists() and GOLD_TAXONOMY.exists() and not args.force:
        sys.exit(f"{GOLD_LABELS.name} exists: the taxonomy is frozen once labelling starts (--force to override)")
    cur = json.loads(cur_path.read_text(encoding="utf-8"))
    meta = json.loads((OUT / "clusters.json").read_text(encoding="utf-8"))
    k = meta["k"]

    intents = [i for i in cur.get("intents", []) if any(str(i.get(f, "")).strip() for f in ("id", "name", "definition"))]
    ids = [str(i.get("id", "")).strip() for i in intents]
    errors, warnings = [], []
    for i, iid in zip(intents, ids):
        if not re.fullmatch(r"[a-z][a-z0-9_]*", iid):
            errors.append(f"intent id {iid!r} must be snake_case")
        for f in ("name", "definition"):
            if not str(i.get(f, "")).strip():
                errors.append(f"intent {iid or '?'}: missing {f}")
    dupes = {x for x in ids if ids.count(x) > 1}
    if dupes:
        errors.append(f"duplicate intent ids: {sorted(dupes)}")
    if not 2 <= len(intents) <= TOOL_KEY_LIMIT:
        errors.append(f"{len(intents)} intents; the labelling tool supports 2-{TOOL_KEY_LIMIT}")
    elif not 8 <= len(intents) <= 10:
        warnings.append(f"{len(intents)} intents; the design targets 8-10")
    if not any("other" in x for x in ids):
        warnings.append("no catch-all intent (an id containing 'other') - the design keeps one")

    mapping = {}
    for c in range(k):
        v = str(cur.get("clusters", {}).get(str(c), {}).get("intent", "")).strip()
        if v not in ids:
            errors.append(f"cluster {c}: intent {v!r} is not one of your intent ids")
        mapping[c] = v
    unused = sorted(set(ids) - set(mapping.values()))
    if unused:
        warnings.append(f"intents with no cluster (fine for rare ones, e.g. safety): {unused}")

    for w in warnings:
        print("warning:", w)
    if errors:
        print("\n".join("error: " + e for e in errors))
        sys.exit(f"{len(errors)} error(s); nothing written")

    shares = {iid: 0.0 for iid in ids}
    for c in meta["clusters"]:
        shares[mapping[c["cluster"]]] += c["share"]
    GOLD_TAXONOMY.parent.mkdir(parents=True, exist_ok=True)
    GOLD_TAXONOMY.write_text(json.dumps({
        "_doc": "Curated intent taxonomy. Written by src/taxonomy.py finalize from artifacts/taxonomy/curation.json.",
        "created": time.strftime("%Y-%m-%d"),
        "intents": [{"id": iid, "name": str(i["name"]).strip(), "definition": str(i["definition"]).strip(),
                     "boundary_notes": str(i.get("boundary_notes", "")).strip()} for i, iid in zip(intents, ids)],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "cluster_to_intent.json").write_text(json.dumps(
        {"k": k, "embedding": meta["run_info"]["models"]["embedding"]["spec"],
         "mapping": {str(c): v for c, v in mapping.items()}}, indent=2), encoding="utf-8")
    print(f"wrote {GOLD_TAXONOMY} with {len(intents)} intents:")
    for iid in ids:
        print(f"  {iid:28s} {shares[iid]:6.1%} of the taxonomy sample")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("propose", help="cluster pool messages and write a report for curation")
    llm.add_mode_args(p)
    p.add_argument("--naming", choices=["auto", "llm", "manual"], default="auto",
                   help="auto: use the namer model if available, else leave names blank")
    p.add_argument("--k", type=int, help="override the chosen number of clusters")
    f = sub.add_parser("finalize", help="validate curation.json and write data/golden/taxonomy.json")
    f.add_argument("--force", action="store_true", help="overwrite even after labelling has started")
    a = ap.parse_args()
    propose(a) if a.cmd == "propose" else finalize(a)


if __name__ == "__main__":
    main()
