"""Brand scan: choose the brand empirically instead of by vibes.

For every support handle in twcs.csv we measure:
  - volume            : how much data we'd have to work with
  - customers         : distinct customers served (breadth of situations)
  - dm_deflect_rate   : % of brand replies that are just "please DM us"
                        -> a high rate makes reply-drafting degenerate
  - link_only_rate    : % of replies that are essentially a bare URL
  - median_reply_len  : proxy for how substantive replies are
  - lexical_diversity : distinct-word ratio over a reply sample
                        -> low means the brand is running canned macros
  - median_thread_len : multi-turn depth (context available for grounding)
"""
import re, sys, pandas as pd, numpy as np

SRC = "data/raw/twcs.csv"
COLS = ["tweet_id", "author_id", "inbound", "text", "in_response_to_tweet_id"]

print("loading...", flush=True)
df = pd.read_csv(SRC, usecols=COLS, dtype={"author_id": "string", "text": "string"})
print(f"rows={len(df):,}", flush=True)

# In this dataset customers are anonymised numeric ids; brands keep their handle.
df["is_brand"] = ~df["author_id"].str.fullmatch(r"\d+").fillna(False)

DM = re.compile(
    r"\b(dm(?:s|ing|'d)?|direct message|private message|pm us|inbox us)\b", re.I)
LINK_ONLY = re.compile(r"^\s*(@\w+\s+)*https?://\S+\s*$", re.I)
MENTION = re.compile(r"@\w+")
URL = re.compile(r"https?://\S+")

brand = df[df["is_brand"]]
rows = []
for handle, g in brand.groupby("author_id", observed=True):
    n = len(g)
    if n < 2000:                      # need enough history to retrieve from
        continue
    txt = g["text"].dropna()
    # strip mentions/urls before measuring length so "@cust http://..." isn't
    # counted as substance
    body = txt.str.replace(MENTION, "", regex=True).str.replace(URL, "", regex=True).str.strip()
    samp = txt.sample(min(4000, len(txt)), random_state=0)
    words = " ".join(samp.tolist()).lower().split()
    rows.append({
        "brand": handle,
        "volume": n,
        "dm_deflect_rate": float(txt.str.contains(DM, na=False).mean()),
        "link_only_rate": float(txt.str.match(LINK_ONLY, na=False).mean()),
        "median_reply_len": float(body.str.len().median()),
        "lexical_diversity": len(set(words)) / max(len(words), 1),
    })

scan = pd.DataFrame(rows)

# thread depth per brand: count tweets in each conversation the brand took part in
print("thread depth...", flush=True)
parent = dict(zip(df["tweet_id"], df["in_response_to_tweet_id"]))
author = dict(zip(df["tweet_id"], df["author_id"]))
depths = {}
for tid in brand["tweet_id"].sample(min(60000, len(brand)), random_state=0):
    d, cur, seen = 0, tid, set()
    while cur == cur and cur is not None and cur not in seen and d < 25:
        seen.add(cur); d += 1
        cur = parent.get(cur)
        if pd.isna(cur): break
    depths.setdefault(author[tid], []).append(d)
scan["median_thread_depth"] = scan["brand"].map(
    {k: float(np.median(v)) for k, v in depths.items()})

# A brand is attractive when it is high-volume AND actually resolves in-channel.
scan["resolves_in_channel"] = 1 - scan["dm_deflect_rate"] - scan["link_only_rate"]
scan = scan.sort_values("volume", ascending=False)

pd.set_option("display.width", 200, "display.max_rows", 60)
out = scan.head(40).round(3)
print(out.to_string(index=False))
scan.to_csv("artifacts/brand_scan.csv", index=False)
print("\nwrote artifacts/brand_scan.csv", flush=True)
