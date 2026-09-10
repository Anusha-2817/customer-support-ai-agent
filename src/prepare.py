"""Build a clean, leakage-safe dataset of (customer message -> BA reply) cases.

Raw twcs.csv is a flat tweet table. Turning it into support *cases* needs four
non-obvious steps, each of which changes the numbers downstream:

1. Thread reconstruction. Tweets link via in_response_to_tweet_id. We walk each
   chain to its root so a case carries the conversation context that preceded it.

2. Multi-part reply merge. BA (like most brands here) splits long answers across
   tweets marked "1/2", "2/2", or simply chains outbound tweets back-to-back.
   Treating each fragment as its own reply would truncate half the answers and
   make the brand look less informative than it is.

3. Cleaning. Customer @mentions are anonymised numeric ids (noise), URLs carry no
   retrievable content, HTML entities are un-escaped in the raw file (&gt; etc),
   and BA agents sign off with initials (*ALA, ^AZ, /CG) which is style not content.

4. TEMPORAL split. The retrieval pool must strictly precede the evaluation cases
   in time. A random split would let the agent retrieve a reply to the very same
   incident (Twitter support has bursts: one cancellation generates hundreds of
   near-identical tweets within hours) and would inflate every reply metric.
"""
from __future__ import annotations
import html, json, re, sys
import pandas as pd

BRAND = "British_Airways"
RAW = "data/raw/twcs.csv"
OUT = "data/processed"

MENTION_NUM = re.compile(r"@\d+\s*")          # anonymised customer handles
URL = re.compile(r"https?://\S+")
# Agent sign-offs, stripped from BA's own tweets only. An inventory of all 29,361
# raw BA tweets (decision log #14) found 81% end in a sign-off, in many shapes:
# ^Jane (17,973), ^R (1,970), ^HP (1,794), ^DaniH (1,089), ^Alex C, ^Lisa.,
# ^Beth S., ^ Barbara, ^jm. In BA tweets "^" is only ever used this way, so the
# caret and the name after it are removed wherever they appear. Left in, retrieved
# exemplars would teach the agent to sign replies as a real employee.
# Customer tweets are NOT touched: 0.18% of them address an agent by name
# ("Thanks ^Kev"), which is real content.
# The name is optional: 66 pool replies ended in a bare "^".
BRAND_SIG = re.compile(r"\s*\^(?:\s?[A-Za-z]+(?:\s[A-Z]\.?(?=\s|$))?\.?)?")
INITIAL_SIG = re.compile(r"\s*[\*/][A-Z]{2,3}\s*$")   # other brands: *ALA, /CG
PART = re.compile(r"\s*\(?\d\s*/\s*\d\)?\s*$")  # "1/2" / "(2/2)" part markers
WS = re.compile(r"\s+")


def clean(text: str, keep_brand_mention: bool = True, brand: bool = False) -> str:
    t = html.unescape(str(text))
    t = URL.sub("<link>", t)
    t = MENTION_NUM.sub("", t)
    if not keep_brand_mention:
        t = re.sub(r"@British_Airways\s*", "", t, flags=re.I)
    t = PART.sub("", t)   # "... ^Linda 1/2": drop the part marker first,
    if brand:             # then the sign-off it was hiding
        t = BRAND_SIG.sub("", t)
        t = INITIAL_SIG.sub("", t)
    return WS.sub(" ", t).strip()


def main(sample_cases: int | None = None) -> None:
    print("loading raw...", flush=True)
    df = pd.read_csv(
        RAW,
        dtype={"author_id": "string", "text": "string",
               "in_response_to_tweet_id": "float64", "response_tweet_id": "string"},
    )
    df["created_at"] = pd.to_datetime(df["created_at"], format="mixed", utc=True)

    # --- restrict to conversations BA took part in -------------------------
    ba_ids = set(df.loc[df.author_id == BRAND, "tweet_id"])
    parent = dict(zip(df.tweet_id, df.in_response_to_tweet_id))
    keep: set[int] = set()
    for tid in ba_ids:                       # walk each BA tweet back to root
        cur, hops = tid, 0
        while cur is not None and not pd.isna(cur) and cur not in keep and hops < 30:
            keep.add(int(cur)); cur = parent.get(int(cur)); hops += 1
    conv = df[df.tweet_id.isin(keep)].copy()
    conv = conv.sort_values("created_at")
    print(f"BA-involved tweets: {len(conv):,}", flush=True)

    idx = conv.set_index("tweet_id")
    is_brand = (conv.author_id == BRAND)

    # --- merge consecutive BA tweets that continue one another -------------
    # A BA tweet whose parent is also a BA tweet is a continuation fragment.
    ba = conv[is_brand]
    ba_parent_is_ba = ba.in_response_to_tweet_id.map(
        lambda p: (not pd.isna(p)) and int(p) in ba_ids)
    heads = ba[~ba_parent_is_ba]              # first fragment of each reply
    children: dict[int, list[int]] = {}
    for t, p in zip(ba.tweet_id, ba.in_response_to_tweet_id):
        if not pd.isna(p):
            children.setdefault(int(p), []).append(int(t))

    def full_reply(tid: int) -> str:
        parts, cur, hops = [], tid, 0
        while cur is not None and hops < 6:
            parts.append(idx.at[cur, "text"])
            nxt = [c for c in children.get(cur, []) if c in ba_ids]
            cur = nxt[0] if nxt else None
            hops += 1
        # clean each fragment first, so an earlier fragment's part marker and
        # sign-off don't survive in the middle of the merged reply
        return clean(" ".join(clean(str(p), brand=True) for p in parts), brand=True)

    # --- build cases -------------------------------------------------------
    rows = []
    for tid, par, ts in zip(heads.tweet_id, heads.in_response_to_tweet_id, heads.created_at):
        if pd.isna(par):
            continue
        par = int(par)
        if par not in idx.index:
            continue
        if idx.at[par, "author_id"] == BRAND:
            continue                          # not a customer-initiated turn
        # conversation context: turns strictly before the customer message
        ctx, cur, hops = [], idx.at[par, "in_response_to_tweet_id"], 0
        while cur is not None and not pd.isna(cur) and hops < 4:
            cur = int(cur)
            if cur not in idx.index:
                break
            who = "brand" if idx.at[cur, "author_id"] == BRAND else "customer"
            ctx.append({"role": who, "text": clean(idx.at[cur, "text"], brand=(who == "brand"))})
            cur = idx.at[cur, "in_response_to_tweet_id"]; hops += 1
        rows.append({
            "case_id": int(par),
            "created_at": ts.isoformat(),
            "context": list(reversed(ctx)),
            "customer_msg": clean(idx.at[par, "text"], keep_brand_mention=False),
            "brand_reply": full_reply(int(tid)),
        })

    cases = pd.DataFrame(rows).sort_values("created_at").reset_index(drop=True)
    # drop empties and near-duplicate customer messages (burst events)
    cases = cases[(cases.customer_msg.str.len() > 15) & (cases.brand_reply.str.len() > 10)]
    cases["dedup_key"] = cases.customer_msg.str.lower().str.replace(r"[^a-z ]", "", regex=True)
    before = len(cases)
    cases = cases.drop_duplicates("dedup_key").drop(columns="dedup_key")
    print(f"cases: {len(cases):,} (dropped {before-len(cases):,} near-dupes)", flush=True)

    # --- temporal split ----------------------------------------------------
    cut = int(len(cases) * 0.75)
    pool, evalset = cases.iloc[:cut], cases.iloc[cut:]
    print(f"pool  : {len(pool):,}  {pool.created_at.min()[:10]} -> {pool.created_at.max()[:10]}")
    print(f"eval  : {len(evalset):,}  {evalset.created_at.min()[:10]} -> {evalset.created_at.max()[:10]}")

    pool.to_json(f"{OUT}/pool.jsonl", orient="records", lines=True)
    evalset.to_json(f"{OUT}/eval_universe.jsonl", orient="records", lines=True)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
