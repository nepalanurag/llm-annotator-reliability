"""Gemini text-classification client.

Uses the stored `custom.google-gemini` connector credential via the dynamic
credential surrogate helper. Authenticated requests go only to
generativelanguage.googleapis.com. Raw credentials are never printed, logged,
or persisted.

Each call asks the model to classify one abstract into a fixed label set and
to return a confidence in [0,1] as strict JSON. Every raw API response is
cached to disk before parsing so a run can be resumed and audited.
"""
import hashlib
import json
import os
import time
import urllib.request
import urllib.error
import sys

sys.path.insert(0, "/opt/hatch/skills/skill-creator/bin")
from dynamic_credentials import (  # noqa: E402
    add_surrogate_to_request, read_response_body)

CREDENTIAL = "custom.google-gemini"
HOST = "generativelanguage.googleapis.com"
MODEL = "models/gemini-3.5-flash-lite"
BASE_URL = f"https://{HOST}/v1beta"

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "api_cache")
REQUEST_DELAY_S = 3.0


def _authed_request(path, body):
    url = BASE_URL + path
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    add_surrogate_to_request(req, CREDENTIAL, allowed_hosts=[HOST])
    return req


def _cache_path(cache_key):
    digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, digest + ".json")


def classify(text, labels, cache_key=None, max_retries=4, temperature=0.0):
    """Classify `text` into one of `labels`.

    Returns (result_dict, usage_dict) where result_dict has keys
    'label' and 'confidence', and usage_dict has
    'prompt_tokens' / 'output_tokens'. On unparseable output after all
    retries, result_dict is {'label': None, 'confidence': None} and the raw
    text is kept under 'raw'.
    """
    labels_str = "[" + ", ".join(labels) + "]"
    prompt = (
        "Classify the scientific abstract below into exactly one of these "
        f"topic labels: {labels_str}.\n"
        "Consider the main disease the abstract is about.\n"
        "Reply with ONLY a JSON object with two keys: \"label\" (one of the "
        "topic labels exactly) and \"confidence\" (your probability that the "
        "label is correct, a number between 0 and 1).\n\nAbstract:\n" + text
    )
    cache_key = cache_key or ("classify:" + labels_str + ":" + text)
    cpath = _cache_path(cache_key)
    if os.path.exists(cpath):
        with open(cpath, encoding="utf-8") as f:
            raw = json.load(f)
    else:
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "maxOutputTokens": 64,
                "temperature": temperature,
            },
        }
        last_err = None
        for attempt in range(max_retries):
            try:
                time.sleep(REQUEST_DELAY_S)
                req = _authed_request(f"/{MODEL}:generateContent", body)
                with urllib.request.urlopen(req, timeout=90) as resp:
                    raw = json.loads(read_response_body(resp))
                break
            except (urllib.error.URLError, urllib.error.HTTPError,
                    TimeoutError, json.JSONDecodeError) as e:
                last_err = e
                wait = 2 ** attempt * 2.0
                time.sleep(wait)
        else:
            raw = {"_client_error": f"{type(last_err).__name__}: {last_err}"}
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(cpath, "w", encoding="utf-8") as f:
            json.dump(raw, f)

    # Parse defensively
    result = {"label": None, "confidence": None, "raw": None}
    usage = {"prompt_tokens": 0, "output_tokens": 0}
    um = raw.get("usageMetadata", {}) if isinstance(raw, dict) else {}
    usage["prompt_tokens"] = int(um.get("promptTokenCount", 0) or 0)
    usage["output_tokens"] = int(um.get("candidatesTokenCount", 0) or 0)
    try:
        parts = raw["candidates"][0]["content"]["parts"]
        text_out = next(p["text"] for p in parts if "text" in p)
        result["raw"] = text_out
        parsed = json.loads(text_out)
        lab = str(parsed.get("label", "")).strip().lower()
        conf = parsed.get("confidence", None)
        result["label"] = lab if lab in [str(x).lower() for x in labels] else None
        try:
            conf = float(conf)
            result["confidence"] = conf if 0.0 <= conf <= 1.0 else None
        except (TypeError, ValueError):
            result["confidence"] = None
    except Exception:
        pass
    return result, usage
