# Adversarial annotation report

Mode: dry-run | Backend under test: rule | Cases: 52

## What this measures

Hard cases target the weakest agreement categories from the repo's error analysis (REPORT.md 4.5): the diabetes/hypertension boundary, drug-name traps, and symptom overlap — plus sarcastic reviews for the reviews axis. In --dry-run mode the cases are rule-generated templates and the backend under test is the keyword baseline, so this report measures the *baseline's* brittleness, not the LLM's. That is an honest proxy: it tells you which trap types a naive annotator falls for. To measure the real annotator, run with --llm (needs the custom.google-gemini connector credential).

## Per-trap-type agreement

| Trap type | n | Correct | Accuracy |
|---|---|---|---|
| boundary_comorbidity | 8 | 8 | 1.000 |
| drug_name_trap | 8 | 8 | 1.000 |
| symptom_overlap | 8 | 4 | 0.500 |
| ambiguous_multi_disease | 8 | 4 | 0.500 |
| hedging_language | 8 | 8 | 1.000 |
| sarcastic_review | 12 | 4 | 0.333 |

Overall: 36/52 = 0.692

## Weakest trap: sarcastic_review (accuracy 0.333)

- adv-046: intended **negative**, predicted **positive** (conf 0.5). Praise words used ironically throughout; true label is negative.
- adv-043: intended **negative**, predicted **positive** (conf 0.8808). Sarcastic positive surface sentiment with negative intent; keyword annotators count 'love' and 'five stars'.
- adv-047: intended **negative**, predicted **positive** (conf 0.5). Praise words used ironically throughout; true label is negative.
- adv-040: intended **negative**, predicted **positive** (conf 0.8808). Sarcastic positive surface sentiment with negative intent; keyword annotators count 'love' and 'five stars'.
- adv-042: intended **negative**, predicted **positive** (conf 0.8808). Sarcastic positive surface sentiment with negative intent; keyword annotators count 'love' and 'five stars'.
- adv-045: intended **negative**, predicted **positive** (conf 0.5). Praise words used ironically throughout; true label is negative.

## Reading the misses

Trap types that broke this backend (accuracy < 0.75): sarcastic_review (0.33), ambiguous_multi_disease (0.50), symptom_overlap (0.50). These are the blind spots to fix first: for sarcastic reviews the backend needs pragmatic (not lexical) cues; for symptom overlap and multi-disease ambiguity it needs the v2 prompt's tie-break rule ('primary disease under study') or a stronger model.
Trap types this backend survived (accuracy >= 0.75): hedging_language (1.00), boundary_comorbidity (1.00), drug_name_trap (1.00). Note this is backend-specific: the boundary and drug-name traps are designed around the failure mode the repo's error analysis found in the *LLM* (9 of 10 errors on the diabetes/hypertension boundary). The keyword backend survives them here only because the templates mention the true disease's vocabulary more often. Run with --llm to measure the real annotator on these traps.

## Case inventory

| case_id | trap_type | intended | generator |
|---|---|---|---|
| adv-046 | sarcastic_review | negative | dry-run-templates |
| adv-037 | hedging_language | hypertension | dry-run-templates |
| adv-043 | sarcastic_review | negative | dry-run-templates |
| adv-003 | boundary_comorbidity | diabetes | dry-run-templates |
| adv-010 | drug_name_trap | hypertension | dry-run-templates |
| adv-047 | sarcastic_review | negative | dry-run-templates |
| adv-039 | hedging_language | hypertension | dry-run-templates |
| adv-049 | sarcastic_review | positive | dry-run-templates |
| adv-028 | ambiguous_multi_disease | migraine | dry-run-templates |
| adv-050 | sarcastic_review | positive | dry-run-templates |
| adv-016 | symptom_overlap | diabetes | dry-run-templates |
| adv-018 | symptom_overlap | diabetes | dry-run-templates |
| adv-015 | drug_name_trap | diabetes | dry-run-templates |
| adv-017 | symptom_overlap | diabetes | dry-run-templates |
| adv-032 | hedging_language | diabetes | dry-run-templates |
| adv-035 | hedging_language | diabetes | dry-run-templates |
| adv-023 | symptom_overlap | hypertension | dry-run-templates |
| adv-021 | symptom_overlap | hypertension | dry-run-templates |
| adv-022 | symptom_overlap | hypertension | dry-run-templates |
| adv-027 | ambiguous_multi_disease | asthma | dry-run-templates |
| adv-019 | symptom_overlap | diabetes | dry-run-templates |
| adv-012 | drug_name_trap | diabetes | dry-run-templates |
| adv-051 | sarcastic_review | positive | dry-run-templates |
| adv-014 | drug_name_trap | diabetes | dry-run-templates |
| adv-048 | sarcastic_review | positive | dry-run-templates |
| adv-008 | drug_name_trap | hypertension | dry-run-templates |
| adv-040 | sarcastic_review | negative | dry-run-templates |
| adv-033 | hedging_language | diabetes | dry-run-templates |
| adv-025 | ambiguous_multi_disease | asthma | dry-run-templates |
| adv-036 | hedging_language | hypertension | dry-run-templates |
| adv-005 | boundary_comorbidity | hypertension | dry-run-templates |
| adv-034 | hedging_language | diabetes | dry-run-templates |
| adv-042 | sarcastic_review | negative | dry-run-templates |
| adv-004 | boundary_comorbidity | hypertension | dry-run-templates |
| adv-000 | boundary_comorbidity | diabetes | dry-run-templates |
| adv-030 | ambiguous_multi_disease | migraine | dry-run-templates |
| adv-026 | ambiguous_multi_disease | asthma | dry-run-templates |
| adv-045 | sarcastic_review | negative | dry-run-templates |
| adv-044 | sarcastic_review | negative | dry-run-templates |
| adv-041 | sarcastic_review | negative | dry-run-templates |
| adv-002 | boundary_comorbidity | diabetes | dry-run-templates |
| adv-006 | boundary_comorbidity | hypertension | dry-run-templates |
| adv-020 | symptom_overlap | hypertension | dry-run-templates |
| adv-029 | ambiguous_multi_disease | migraine | dry-run-templates |
| adv-038 | hedging_language | hypertension | dry-run-templates |
| adv-024 | ambiguous_multi_disease | asthma | dry-run-templates |
| adv-007 | boundary_comorbidity | hypertension | dry-run-templates |
| adv-011 | drug_name_trap | hypertension | dry-run-templates |
| adv-013 | drug_name_trap | diabetes | dry-run-templates |
| adv-009 | drug_name_trap | hypertension | dry-run-templates |
| adv-031 | ambiguous_multi_disease | migraine | dry-run-templates |
| adv-001 | boundary_comorbidity | diabetes | dry-run-templates |
