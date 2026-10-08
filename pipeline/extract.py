"""Extract stage: fetch abstracts from the arXiv API (or a local CSV).

arXiv API: http://export.arxiv.org/api/query — keyless, Atom feed. Polite:
3s delay between requests, bounded retries with exponential backoff, clear
errors on bad queries. Output is a DataFrame validated against
pipeline.schemas.RawAbstract.

The --from-csv path loads the repo's data/abstracts.csv (columns pmid, topic,
text) so the pipeline runs fully offline and reproduces the published corpus.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET

import pandas as pd

from pipeline.config import load_settings
from pipeline.logging import configure_logging, get_logger
from pipeline.schemas import validate_or_raise, RawAbstract

log = get_logger(__name__)

ARXIV_API = "http://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom"}


def _fetch(url: str, timeout_s: float, max_retries: int) -> bytes:
    last: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "llm-annotator-pipeline/0.1"}
            )
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"arXiv API returned HTTP {resp.status}")
                return resp.read()
        except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            last = exc
            wait = 2.0 * (2**attempt)
            log.warning(
                "arxiv_fetch_retry", attempt=attempt, wait_s=wait, error=str(exc)
            )
            time.sleep(wait)
    raise RuntimeError(f"arXiv API fetch failed after {max_retries} retries: {last}")


def extract_arxiv(
    query: str,
    max_results: int = 50,
    timeout_s: float = 30.0,
    max_retries: int = 4,
    polite_delay_s: float = 3.0,
) -> pd.DataFrame:
    """Fetch up to max_results abstracts from arXiv for `query`."""
    if not query or not query.strip():
        raise ValueError("arXiv query must not be empty")
    if max_results < 1:
        raise ValueError("max_results must be >= 1")
    params = urllib.parse.urlencode(
        {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submitted",
            "sortOrder": "descending",
        }
    )
    raw = _fetch(f"{ARXIV_API}?{params}", timeout_s, max_retries)
    time.sleep(polite_delay_s)
    root = ET.fromstring(raw)
    rows = []
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    for entry in root.findall("atom:entry", NS):
        arxiv_id = entry.findtext("atom:id", namespaces=NS, default="").rsplit("/", 1)[
            -1
        ]
        title = (entry.findtext("atom:title", namespaces=NS, default="") or "").strip()
        summary = (
            entry.findtext("atom:summary", namespaces=NS, default="") or ""
        ).strip()
        if not arxiv_id or not summary:
            continue
        rows.append(
            {
                "doc_id": f"arxiv:{arxiv_id}",
                "title": title,
                "text": summary,
                "source": "arxiv",
                "retrieved_at": now,
            }
        )
    df = pd.DataFrame(
        rows, columns=["doc_id", "title", "text", "source", "retrieved_at"]
    )
    log.info("arxiv_extract_complete", query=query, n=len(df))
    return validate_or_raise(RawAbstract, df)


def extract_csv(path: str) -> pd.DataFrame:
    """Load the repo's data/abstracts.csv (pmid, topic, text) as raw abstracts."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"CSV not found: {path}. Expected the repo's data/abstracts.csv "
            "with columns pmid, topic, text."
        )
    df = pd.read_csv(path)
    missing = {"pmid", "text"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV {path} missing columns: {sorted(missing)}")
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    out = pd.DataFrame(
        {
            "doc_id": "pmid:" + df["pmid"].astype(str),
            "title": df.get("topic", ""),
            "text": df["text"].astype(str),
            "source": "csv",
            "retrieved_at": now,
            "ref_label": df["topic"].astype(str) if "topic" in df.columns else None,
        }
    )
    out = out[out["text"].str.strip().ne("")]
    log.info("csv_extract_complete", path=path, n=len(out))
    return validate_or_raise(RawAbstract, out.reset_index(drop=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract abstracts (arXiv API or local CSV)."
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--from-csv", metavar="PATH", help="Load abstracts from a CSV file"
    )
    src.add_argument(
        "--query", default=None, help="arXiv search query (default: config)"
    )
    parser.add_argument("--max-results", type=int, default=None)
    parser.add_argument("--out", required=True, help="Output parquet path")
    args = parser.parse_args(argv)
    settings = load_settings()
    configure_logging(settings.log_format, settings.log_level)

    if args.from_csv:
        df = extract_csv(args.from_csv)
    else:
        df = extract_arxiv(
            args.query or settings.arxiv_query,
            max_results=args.max_results or settings.max_results,
            timeout_s=settings.request_timeout_s,
            max_retries=settings.max_retries,
            polite_delay_s=settings.polite_delay_s,
        )
    if df.empty:
        print("error: extraction returned zero abstracts", file=sys.stderr)
        return 2
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"extracted {len(df)} abstracts -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
