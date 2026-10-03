# dashboard-data: llm-annotator-reliability

Key results from `notebooks/analysis.ipynb` ("Can an LLM replace human
annotators?"), exported for the interactive dashboard. All numbers come from
the notebook's executed outputs and the repo's `results.json`. 248 PubMed
abstracts labeled zero-shot by Gemini 3.5 Flash-Lite; labels are distant
supervision (the PubMed query that retrieved each abstract), not
hand-verified gold.

## Files

- **metrics.json** — headline numbers.
  Fields: `n`, `n_correct`, `model`, `seed`, `accuracy` + `accuracy_ci95`
  (Wilson 95% interval), `cohens_kappa` + `cohens_kappa_ci95`,
  `gwets_ac1` + `gwets_ac1_ci95` (2,000-resample bootstrap percentile
  intervals), `ece` (expected calibration error).
- **per_class.json** — per-topic precision/recall/F1.
  Fields: `classes[]`: `label`, `precision`, `recall`, `f1`, `support`.
- **calibration_bins.json** — confidence calibration deciles.
  Fields: `bins[]`: `lo`/`hi` (confidence bin edges), `n`,
  `mean_confidence`, `accuracy` (observed fraction correct).
- **mcnemar.json** — LLM vs supervised baseline, paired test.
  Fields: `test_n`, `baseline`, `baseline_accuracy`,
  `llm_accuracy_same_test`, `discordant_baseline_only_right`,
  `discordant_llm_only_right`, `mcnemar_p` (exact binomial p).
- **error_analysis.json** — accuracy by abstract length and by topic.
  Fields: `by_length[]`: `quartile`, `accuracy`, `n`, `ci95` (Wilson);
  `by_topic[]`: `topic`, `accuracy`, `n`.
- **cost.json** — measured vs estimated annotation cost.
  Fields: `n_annotations`, `input_tokens_total`, `output_tokens_total`,
  `price_input_per_1m_usd`, `price_output_per_1m_usd`,
  `llm_cost_usd_total`, `llm_cost_per_1000_usd`,
  `human_cost_per_item_usd_estimate`, `human_cost_per_1000_usd_estimate`.
- **label_distribution.json** — true vs predicted label counts.
  Fields: `true_topics`, `llm_predicted`, `parse_failures`.
