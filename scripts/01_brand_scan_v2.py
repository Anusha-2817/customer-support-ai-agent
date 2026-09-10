"""Brand scan v2.

v1 only counted 'DM us' as deflection and ranked AmazonHelp top with a 0.6%
deflection rate, which looked wrong. Inspecting samples showed Amazon deflects
to a *web form* instead. So deflection is generalised here into three channels,
and we add a positive signal (does the reply contain concrete resolution
language?) rather than only penalties.
"""
import re, pandas as pd, numpy as np

CANDIDATES = ["AmazonHelp","AppleSupport","Uber_Support","SpotifyCares","Delta","Tesco",
              "AmericanAir","British_Airways","SouthwestAir","VirginTrains","XboxSupport",
              "hulu_support","sainsburys","GWRHelp","AskPlayStation","ChipotleTweets",
              "VerizonSupport","AskAmex","MicrosoftHelps","AdobeCare","marksandspencer",
              "SW_Help","BofA_Help","AirAsiaSupport","ATVIAssist"]

df = pd.read_csv("data/raw/twcs.csv", usecols=["tweet_id","author_id","inbound","text"],
                 dtype={"author_id":"string","text":"string"})
df = df[df["author_id"].isin(CANDIDATES)]

DM      = re.compile(r"\b(?:dm|dms|dming|direct message|private message|pm us|inbox us)\b", re.I)
# "click here / contact us at <url>" -- deflection to a web channel
WEBFORM = re.compile(r"(contact us|reach out|get in touch|report it|more (?:help|info)|"
                     r"details here|please visit|see|via|through|use|try)\b[^.]{0,40}https?://", re.I)
HASLINK = re.compile(r"https?://")
PHONE   = re.compile(r"\b(?:call us|give us a call|\d{3}[-.\s]\d{3}[-.\s]\d{4}|1-8\d{2})\b", re.I)
# positive signal: language that carries an actual answer/action
RESOLVE = re.compile(r"\b(?:you can|you'll need to|here's how|try|make sure|check|"
                     r"we've|i've|refund|resend|reset|updated|arrived|scheduled|"
                     r"delayed|rebook|confirm|because|due to|policy|within \d+)\b", re.I)
APOLOGY = re.compile(r"^\s*(@\w+\s*)*(?:i'm |we're |so |very )?(?:sorry|apolog)", re.I)

rows = []
for b, g in df.groupby("author_id", observed=True):
    t = g["text"].dropna()
    dm, web = t.str.contains(DM, na=False), t.str.contains(WEBFORM, na=False)
    ph = t.str.contains(PHONE, na=False)
    deflect = dm | web | ph
    rows.append({
        "brand": b, "volume": len(t),
        "dm": dm.mean(), "web": web.mean(), "phone": ph.mean(),
        "any_deflect": deflect.mean(),
        "has_link": t.str.contains(HASLINK, na=False).mean(),
        "resolve_lang": t.str.contains(RESOLVE, na=False).mean(),
        "apology_open": t.str.contains(APOLOGY, na=False).mean(),
        # the metric that matters: replies that answer instead of redirecting
        "self_contained": (~deflect & t.str.contains(RESOLVE, na=False)).mean(),
    })

scan = pd.DataFrame(rows).sort_values("self_contained", ascending=False)
pd.set_option("display.width", 220)
print(scan.round(3).to_string(index=False))
scan.to_csv("artifacts/brand_scan_v2.csv", index=False)

print("\n" + "="*100)
for b in ["AmazonHelp","Delta","AmericanAir","SpotifyCares","British_Airways"]:
    print(f"\n--- {b} ---")
    for s in df[df.author_id==b]["text"].dropna().sample(6, random_state=7):
        print("   ", s[:190].replace("\n"," "))
