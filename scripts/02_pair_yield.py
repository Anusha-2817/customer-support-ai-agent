"""Tie-breaker: which candidate brand gives the most usable (customer -> brand
reply) pairs, and how varied are the incoming customer messages?

A brand is only useful to us if we can reconstruct supervised pairs: an inbound
customer tweet that has an outbound brand tweet replying to it. Volume of brand
tweets alone overstates this.
"""
import re, pandas as pd
C = ["British_Airways","GWRHelp","Tesco","sainsburys","AirAsiaSupport",
     "SpotifyCares","AmericanAir","SW_Help","MicrosoftHelps","AmazonHelp"]

df = pd.read_csv("data/raw/twcs.csv",
                 usecols=["tweet_id","author_id","inbound","text","in_response_to_tweet_id"],
                 dtype={"author_id":"string","text":"string"})
by_id = df.set_index("tweet_id")
brand_tw = df[df.author_id.isin(C)]

out = []
for b, g in brand_tw.groupby("author_id", observed=True):
    # brand tweets that are replies to something we can look up
    par = g["in_response_to_tweet_id"].dropna()
    joined = par.map(by_id["text"]).dropna()
    par_auth = par.map(by_id["author_id"])
    # keep only pairs where the parent came from a customer (numeric id)
    cust = par_auth.str.fullmatch(r"\d+").fillna(False)
    n_pairs = int(cust.sum())
    cust_txt = par[cust].map(by_id["text"]).dropna()
    words = " ".join(cust_txt.sample(min(3000,len(cust_txt)), random_state=1).str.lower().tolist()).split()
    out.append({"brand": b, "brand_tweets": len(g), "usable_pairs": n_pairs,
                "pair_rate": round(n_pairs/len(g), 3),
                "cust_vocab": len(set(words)),
                "median_cust_len": int(cust_txt.str.len().median())})
res = pd.DataFrame(out).sort_values("usable_pairs", ascending=False)
print(res.to_string(index=False))
res.to_csv("artifacts/pair_yield.csv", index=False)

for b in ["British_Airways","GWRHelp","Tesco"]:
    print(f"\n===== inbound customer messages to {b} =====")
    par = brand_tw[brand_tw.author_id==b]["in_response_to_tweet_id"].dropna()
    txt = par.map(by_id["text"]).dropna()
    for s in txt.sample(10, random_state=3):
        print("  *", re.sub(r"\s+"," ", s)[:180])
