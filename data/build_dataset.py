"""Build the annotation dataset from PubMed via Entrez (no API key, <=3 req/s).

Queries four fixed MeSH terms, fetches title+abstract, labels each record by
the query topic that retrieved it, then draws a stratified random sample of
250 records with a fixed seed so the dataset is reproducible.

Output: data/abstracts.csv with columns
    pmid, topic, title, abstract, text, n_chars, n_words
"""
import csv
import random
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

SEED = 20261002
TOPICS = {
    "diabetes": "diabetes mellitus[MeSH Terms]",
    "hypertension": "hypertension[MeSH Terms]",
    "asthma": "asthma[MeSH Terms]",
    "migraine": "migraine disorders[MeSH Terms]",
}
N_PER_TOPIC = 70          # fetched per topic before sampling
TARGET_TOTAL = 250
BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
COMMON = {"tool": "llm-annotator-reliability", "email": "noreply@example.com"}


def _get(url, timeout=60):
    time.sleep(0.4)  # stay under the 3 req/s Entrez limit
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def search_ids(term, retmax):
    url = BASE + "esearch.fcgi?" + urllib.parse.urlencode(
        {**COMMON, "db": "pubmed", "term": term, "retmax": retmax, "retmode": "json"}
    )
    import json
    d = json.loads(_get(url))
    return d["esearchresult"]["idlist"]


def fetch_records(idlist):
    """Fetch title+abstract for a list of PMIDs. Returns list of dicts."""
    out = []
    for i in range(0, len(idlist), 50):
        chunk = idlist[i:i + 50]
        url = BASE + "efetch.fcgi?" + urllib.parse.urlencode(
            {**COMMON, "db": "pubmed", "id": ",".join(chunk),
             "retmode": "xml", "rettype": "abstract"}
        )
        xml_bytes = _get(url, timeout=120)
        root = ET.fromstring(xml_bytes)
        for art in root.findall(".//PubmedArticle"):
            pmid_el = art.find(".//PMID")
            title_el = art.find(".//ArticleTitle")
            abs_el = art.find(".//Abstract/AbstractText")
            if pmid_el is None or title_el is None or abs_el is None:
                continue
            title = "".join(title_el.itertext()).strip()
            abstract = " ".join("".join(a.itertext()).strip()
                               for a in art.findall(".//Abstract/AbstractText")).strip()
            if len(abstract) < 300:  # need a real abstract, not a stub
                continue
            text = title + "\n" + abstract
            out.append({
                "pmid": pmid_el.text.strip(),
                "title": title,
                "abstract": abstract,
                "text": text,
                "n_chars": len(text),
                "n_words": len(text.split()),
            })
    return out


def main():
    rng = random.Random(SEED)
    rows = []
    for topic, term in TOPICS.items():
        ids = search_ids(term, retmax=200)
        rng.shuffle(ids)  # shuffle before taking so we are not biased to recent papers
        recs = fetch_records(ids[:N_PER_TOPIC + 30])
        recs = recs[:N_PER_TOPIC]
        for rec in recs:
            rec["topic"] = topic
        rows.append((topic, recs))
        print(f"{topic}: {len(recs)} records", flush=True)

    # Stratified sample: equal counts per topic, then shuffle
    per_topic = TARGET_TOTAL // len(TOPICS)
    final = []
    for topic, recs in rows:
        take = min(per_topic, len(recs))
        final.extend(recs[:take])
    rng.shuffle(final)
    # De-duplicate on pmid just in case a record matched two MeSH terms
    seen, dedup = set(), []
    for r in final:
        if r["pmid"] in seen:
            continue
        seen.add(r["pmid"])
        dedup.append(r)

    with open("data/abstracts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pmid", "topic", "title", "abstract",
                                         "text", "n_chars", "n_words"])
        w.writeheader()
        for r in dedup:
            w.writerow({k: r[k] for k in w.fieldnames})
    print(f"Wrote {len(dedup)} records to data/abstracts.csv")


if __name__ == "__main__":
    main()
