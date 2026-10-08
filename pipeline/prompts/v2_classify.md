# Classify prompt v2

Version: v2
Created: 2026-10-08
Change vs v1: adds an explicit tie-break rule for comorbidity mentions. The
repo's error analysis (REPORT.md 4.5) found 9 of 10 errors on the
diabetes/hypertension boundary, typically where both conditions are mentioned.
v2 tells the annotator to pick the PRIMARY disease under study, not the first
one named.
Placeholders: {labels}, {text} (same as v1).

---
Classify the scientific abstract below into exactly one of these topic labels: {labels}.
Consider the main disease the abstract is about. If the abstract mentions more
than one of the listed diseases (for example a comorbidity, a comparison group,
or a drug studied in a different disease), choose the PRIMARY disease the study
is about — the condition of the study population or the outcome being measured —
not the first disease named.
Reply with ONLY a JSON object with two keys: "label" (one of the topic labels exactly) and "confidence" (your probability that the label is correct, a number between 0 and 1).

Abstract:
{text}
