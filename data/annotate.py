"""Annotate every abstract in data/abstracts.csv with the Gemini classifier.

One API call per abstract. Raw responses are cached in data/api_cache/ by
src/gemini_client.py before parsing, so re-runs only parse. Results are
written incrementally to data/annotations.csv so a crash does not lose work.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from gemini_client import classify  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = ["diabetes", "hypertension", "asthma", "migraine"]
OUT = os.path.join(HERE, "annotations.csv")


def main():
    import pandas as pd
    df = pd.read_csv(os.path.join(HERE, "abstracts.csv"))
    done = {}
    if os.path.exists(OUT):
        old = pd.read_csv(OUT)
        # Drop previous parse failures so they get retried.
        old = old[old["parsed_ok"] == 1].copy()
        old.to_csv(OUT, index=False)
        done = dict(zip(old["pmid"].astype(str), old.index))

    rows = []
    for i, row in df.iterrows():
        pmid = str(row["pmid"])
        if pmid in done:
            continue
        result, usage = classify(row["text"], LABELS, cache_key="pmid:" + pmid)
        parsed_ok = result["label"] is not None and result["confidence"] is not None
        rows.append({
            "pmid": pmid,
            "topic": row["topic"],
            "pred_label": result["label"],
            "confidence": result["confidence"],
            "parsed_ok": int(parsed_ok),
            "prompt_tokens": usage["prompt_tokens"],
            "output_tokens": usage["output_tokens"],
        })
        if (len(rows) % 20) == 0:
            print(f"{len(done) + len(rows)}/{len(df)} annotated", flush=True)
    header = ["pmid", "topic", "pred_label", "confidence", "parsed_ok",
              "prompt_tokens", "output_tokens"]
    mode = "a" if os.path.exists(OUT) else "w"
    with open(OUT, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if mode == "w":
            w.writeheader()
        w.writerows(rows)
    print(f"Annotated {len(rows)} new abstracts; total on disk now.")


if __name__ == "__main__":
    main()
