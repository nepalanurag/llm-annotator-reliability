# Disagreement taxonomy report

Mode: dry-run | Disagreements clustered: 50

## Method

Disagreement texts (repo annotation errors + seeded synthetic disagreements) were embedded with TF-IDF and clustered with HDBSCAN. Cluster names in --dry-run mode come from top TF-IDF terms; --llm mode asks Gemini to name and describe each cluster.

## Human-written error analysis (REPORT.md 4.5, the reference)

- **T1_boundary_diabetes_hypertension**: Errors concentrate on the diabetes/hypertension boundary (9 of 10 errors).
- **T2_hypertension_recall**: Hypertension recall collapses (0.855); true hypertension cases labeled as diabetes.
- **T3_comorbidity_mention**: Comorbidity mentions confuse the annotator.
- **T4_drug_name_cue**: Drug names cue the wrong disease (implied by the boundary errors).

## Machine taxonomy

| Cluster | Size | Name | Top terms |
|---|---|---|---|
| -1 | 2 | noise/unclustered | survival, dysfunction, vascular dysfunction, vascular, hsp70, microbial |
| 0 | 4 | cluster-0: concomitant diabetes, concomitant, diabetes present | concomitant diabetes, concomitant, diabetes present, present half, half, half cohort |
| 1 | 7 | cluster-1: participants, albuminuria trial, participants received | participants, albuminuria trial, participants received, renal, renal protection, protection |
| 2 | 4 | cluster-2: outcomes insulin, hypertension management, management improved | outcomes insulin, hypertension management, management improved, improved, improved outcomes, standard |
| 3 | 5 | cluster-3: medications, therapy initiated, initiated blood | medications, therapy initiated, initiated blood, initiated, therapy, unchanged follow |
| 4 | 4 | cluster-4: pressure, ambulatory monitoring, pressure 152 | pressure, ambulatory monitoring, pressure 152, pattern mean, pattern, 24 hour |
| 5 | 6 | cluster-5: risk, 95, ci | risk, 95, ci, 95 ci, health, ckd |
| 6 | 4 | cluster-6: comorbidity burden, disease enrolled, comorbidity | comorbidity burden, disease enrolled, comorbidity, diabetes chronic, burden high, enrolled comorbidity |
| 7 | 6 | cluster-7: control, reduced events, intensive | control, reduced events, intensive, intensive blood, hba1c primary, primary outcome |
| 8 | 8 | cluster-8: systolic pressure, systolic, persisted despite | systolic pressure, systolic, persisted despite, persisted, antihypertensives, antihypertensives systolic |

## Human vs machine agreement

Adjusted Rand index (HDBSCAN vs rule-assigned human themes): **0.184**

Mean cluster purity (majority human theme per cluster): **1.0** over 9 non-noise clusters.

Reading: ARI is chance-corrected and penalizes granularity mismatch — the machine found finer structure (blood-pressure-pattern vs trial-systolic vs intensive-control clusters) than the four human themes, so ARI is modest even when clusters are pure. Purity answers the forgiving question: does each machine cluster mostly contain one human theme? High purity + low ARI means the machine split human themes into subtypes — useful triage, not contradiction. On the real repo errors (n=10, too few to cluster alone), the check is qualitative: boundary/comorbidity/pressure clusters do appear among the top clusters, matching REPORT.md's T1-T4.

## Takeaway

The machine taxonomy is a triage tool, not a replacement for reading the confusion matrix: it surfaces candidate error themes fast, and the ARI score keeps it honest about how well those themes match the analyst's own reading.
