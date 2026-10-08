# Classify prompt v1

Version: v1
Created: 2026-10-08
Source: ported verbatim from the inline prompt in src/gemini_client.py so the
pipeline's first prompt version reproduces the repo's published annotation run.
Placeholders: {labels} (e.g. [diabetes, hypertension, asthma, migraine]),
{text} (the abstract or chunk to classify).

---
Classify the scientific abstract below into exactly one of these topic labels: {labels}.
Consider the main disease the abstract is about.
Reply with ONLY a JSON object with two keys: "label" (one of the topic labels exactly) and "confidence" (your probability that the label is correct, a number between 0 and 1).

Abstract:
{text}
