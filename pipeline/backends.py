"""Pluggable annotation backends.

Two backends ship:

- RuleBackend ("rule"): deterministic keyword scorer. Offline, zero cost, no
  credentials. It is a *weak* annotator on purpose: a cheap baseline that
  exercises the whole pipeline (dedup -> annotate -> agreement -> drift)
  without spending API budget, and a measuring stick for the LLM backends.
- GeminiBackend ("gemini"): calls the repo's published annotator
  (src/gemini_client.classify) with a *versioned* prompt file instead of the
  inline prompt. Needs the custom.google-gemini connector credential at
  runtime; fails loudly and clearly when it is unavailable.

Both return Annotation records validated against pipeline.schemas.Annotation.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Protocol

from pipeline.logging import get_logger

log = get_logger(__name__)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")

LABELS = ["diabetes", "hypertension", "asthma", "migraine"]
REVIEW_LABELS = ["positive", "negative"]


@dataclass(frozen=True)
class Annotation:
    chunk_id: str
    backend: str
    backend_version: str
    model_id: str
    prompt_version: str
    prompt_sha256: str
    label: str | None
    confidence: float | None
    parsed_ok: bool
    latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None


class AnnotationBackend(Protocol):
    name: str
    version: str
    model_id: str

    def annotate(
        self,
        texts: list[tuple[str, str]],
        labels: list[str],
        prompt_version: str = "v1",
    ) -> list[Annotation]:
        """Annotate [(chunk_id, text), ...] -> one Annotation per input."""
        ...


def load_prompt(prompt_version: str) -> tuple[str, str]:
    """Load a versioned prompt file. Returns (template, sha256).

    Prompts are versioned files, never inline strings. Raises FileNotFoundError
    with a clear message for unknown versions.
    """
    path = os.path.join(PROMPTS_DIR, f"{prompt_version}_classify.md")
    if not os.path.exists(path):
        available = sorted(
            f.removesuffix("_classify.md")
            for f in os.listdir(PROMPTS_DIR)
            if f.endswith("_classify.md")
        )
        raise FileNotFoundError(
            f"unknown prompt version {prompt_version!r} (available: {available})"
        )
    with open(path, encoding="utf-8") as f:
        # Prompt files carry a YAML-ish header; the template follows the --- line.
        content = f.read()
    template = (
        content.split("---", 1)[1].strip() if "---" in content else content.strip()
    )
    sha = hashlib.sha256(template.encode("utf-8")).hexdigest()
    return template, sha


# Keyword lists for the rule baseline. Curated from the repo's own vocabulary:
# MeSH-topic abstracts for the four diseases. Deliberately naive — this is the
# cheap baseline, not a competitor.
_KEYWORDS: dict[str, list[str]] = {
    "diabetes": [
        "diabetes",
        "diabetic",
        "insulin",
        "glucose",
        "hba1c",
        "glycemic",
        "hyperglycemia",
        "metformin",
        "pancreatic",
        "type 1",
        "type 2",
        "t1d",
        "t2d",
    ],
    "hypertension": [
        "hypertension",
        "hypertensive",
        "blood pressure",
        "antihypertensive",
        "systolic",
        "diastolic",
        "mmhg",
    ],
    "asthma": [
        "asthma",
        "asthmatic",
        "bronchial",
        "inhaler",
        "wheezing",
        "airway hyperresponsiveness",
        "corticosteroid inhal",
    ],
    "migraine": [
        "migraine",
        "headache",
        "aura",
        "triptan",
        "photophobia",
        "cephalgia",
    ],
    "positive": [
        "love",
        "great",
        "excellent",
        "amazing",
        "perfect",
        "wonderful",
        "best",
        "recommend",
        "happy",
        "satisfied",
        "five star",
        "5 star",
    ],
    "negative": [
        "hate",
        "terrible",
        "awful",
        "worst",
        "broken",
        "disappointed",
        "return",
        "refund",
        "waste",
        "poor",
        "one star",
        "1 star",
    ],
}

# Sarcasm/negation cues the rule backend explicitly cannot handle well.
# Documented as a known weakness (see adversarial_report.md).
_NEGATIONS = {"not", "no", "never", "n't", "barely", "hardly"}


class RuleBackend:
    """Deterministic keyword-scoring annotator. Offline, $0 per annotation."""

    name = "rule"
    version = "1.0"
    model_id = "rule-keyword-scorer/1.0"

    def _score(self, text: str, labels: list[str]) -> dict[str, float]:
        lowered = text.lower()
        scores: dict[str, float] = {}
        for label in labels:
            kws = _KEYWORDS.get(label, [])
            hits = sum(
                len(re.findall(r"\b" + re.escape(kw) + r"\b", lowered)) for kw in kws
            )
            scores[label] = float(hits)
        return scores

    def annotate(
        self,
        texts: list[tuple[str, str]],
        labels: list[str],
        prompt_version: str = "v1",
    ) -> list[Annotation]:
        if prompt_version not in ("v1", "v2"):
            # The rule backend ignores prompt wording, but the version is still
            # recorded for experiment comparability.
            log.warning(
                "rule_backend_ignores_prompt_text", prompt_version=prompt_version
            )
        _, prompt_sha = load_prompt("v1")  # validate prompt exists
        out: list[Annotation] = []
        for chunk_id, text in texts:
            t0 = time.perf_counter()
            if not text or not text.strip():
                out.append(
                    Annotation(
                        chunk_id,
                        self.name,
                        self.version,
                        self.model_id,
                        prompt_version,
                        prompt_sha,
                        None,
                        0.0,
                        False,
                        (time.perf_counter() - t0) * 1000.0,
                    )
                )
                continue
            scores = self._score(text, labels)
            total = sum(scores.values())
            if total == 0:
                # No keyword hit: abstain rather than guess.
                label, conf, ok = None, 0.0, False
            else:
                best = max(scores, key=lambda k: scores[k])
                # Softmax over scores -> confidence in [0,1].
                exps = {k: math.exp(v) for k, v in scores.items()}
                conf = exps[best] / sum(exps.values())
                label, ok = best, True
            out.append(
                Annotation(
                    chunk_id=chunk_id,
                    backend=self.name,
                    backend_version=self.version,
                    model_id=self.model_id,
                    prompt_version=prompt_version,
                    prompt_sha256=prompt_sha,
                    label=label,
                    confidence=round(conf, 4),
                    parsed_ok=ok,
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                )
            )
        log.info(
            "rule_backend_annotated",
            n=len(out),
            abstentions=sum(1 for a in out if a.label is None),
        )
        return out


class GeminiBackend:
    """Repo's published Gemini annotator, driven by a versioned prompt file.

    Sends the versioned prompt text (not an inline string) to
    models/gemini-3.5-flash-lite, mirroring the auth and caching pattern of
    the repo's src/gemini_client.py. Requires the custom.google-gemini
    connector credential at runtime; fails loudly and clearly when it is
    unavailable — never fails silently, never retries forever.
    """

    name = "gemini"
    version = "1.0"
    model_id = "models/gemini-3.5-flash-lite"
    _credential = "custom.google-gemini"
    _host = "generativelanguage.googleapis.com"

    def _call(self, prompt: str, cache_key: str) -> tuple[dict, dict]:
        """POST one prompt; raw response cached to disk before parsing."""
        import hashlib as _hashlib
        import json as _json
        import urllib.request as _urlreq
        import urllib.error as _urlerr

        sys.path.insert(0, "/opt/hatch/skills/skill-creator/bin")
        try:
            from dynamic_credentials import (  # noqa: E402
                add_surrogate_to_request,
                read_response_body,
            )
        except ImportError as exc:
            raise RuntimeError(
                "GeminiBackend needs the custom.google-gemini connector "
                "credential (via the dynamic_credentials surrogate helper). "
                "On this machine it lives under "
                "/opt/hatch/skills/skill-creator/bin; elsewhere, configure the "
                "connector the way the repo README describes for src/gemini_client.py."
            ) from exc

        cache_dir = os.path.join(REPO_ROOT, "data", "api_cache")
        cpath = os.path.join(
            cache_dir, _hashlib.sha256(cache_key.encode()).hexdigest() + ".json"
        )
        if os.path.exists(cpath):
            with open(cpath, encoding="utf-8") as f:
                raw = _json.load(f)
        else:
            body = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "maxOutputTokens": 64,
                    "temperature": 0.0,
                },
            }
            url = f"https://{self._host}/v1beta/{self.model_id}:generateContent"
            req = _urlreq.Request(
                url,
                data=_json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            add_surrogate_to_request(req, self._credential, allowed_hosts=[self._host])
            try:
                with _urlreq.urlopen(req, timeout=90) as resp:
                    raw = _json.loads(read_response_body(resp))
            except (_urlerr.URLError, _urlerr.HTTPError, TimeoutError) as exc:
                raise RuntimeError(
                    f"Gemini API call failed: {type(exc).__name__}: {exc}"
                ) from exc
            os.makedirs(cache_dir, exist_ok=True)
            with open(cpath, "w", encoding="utf-8") as f:
                _json.dump(raw, f)

        usage = {"prompt_tokens": 0, "output_tokens": 0}
        um = raw.get("usageMetadata", {}) if isinstance(raw, dict) else {}
        usage["prompt_tokens"] = int(um.get("promptTokenCount", 0) or 0)
        usage["output_tokens"] = int(um.get("candidatesTokenCount", 0) or 0)
        return raw, usage

    @staticmethod
    def _parse(raw: dict, labels: list[str]) -> tuple[str | None, float | None]:
        try:
            parts = raw["candidates"][0]["content"]["parts"]
            text_out = next(p["text"] for p in parts if "text" in p)
            import json as _json

            parsed = _json.loads(text_out)
            lab = str(parsed.get("label", "")).strip().lower()
            conf = float(parsed.get("confidence"))
            label = lab if lab in [str(x).lower() for x in labels] else None
            conf = conf if 0.0 <= conf <= 1.0 else None
            return label, conf
        except Exception:
            return None, None

    def annotate(
        self,
        texts: list[tuple[str, str]],
        labels: list[str],
        prompt_version: str = "v1",
    ) -> list[Annotation]:
        template, prompt_sha = load_prompt(prompt_version)
        labels_str = "[" + ", ".join(labels) + "]"
        out: list[Annotation] = []
        for chunk_id, text in texts:
            t0 = time.perf_counter()
            prompt = template.replace("{labels}", labels_str).replace("{text}", text)
            raw, usage = self._call(
                prompt,
                cache_key=f"pipeline:{prompt_version}:{prompt_sha[:16]}:{chunk_id}",
            )
            label, conf = self._parse(raw, labels)
            ok = label is not None and conf is not None
            out.append(
                Annotation(
                    chunk_id=chunk_id,
                    backend=self.name,
                    backend_version=self.version,
                    model_id=self.model_id,
                    prompt_version=prompt_version,
                    prompt_sha256=prompt_sha,
                    label=label,
                    confidence=conf,
                    parsed_ok=ok,
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                    input_tokens=usage.get("prompt_tokens"),
                    output_tokens=usage.get("output_tokens"),
                )
            )
        log.info(
            "gemini_backend_annotated",
            n=len(out),
            failures=sum(1 for a in out if not a.parsed_ok),
        )
        return out


def get_backend(name: str) -> AnnotationBackend:
    """Return the backend by name, failing loudly on unknown names."""
    if name == "rule":
        return RuleBackend()
    if name == "gemini":
        return GeminiBackend()
    raise ValueError(f"unknown backend {name!r} (choose 'rule' or 'gemini')")
