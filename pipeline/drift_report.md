# Label drift report

Checked at: 2026-10-08T16:53:38.272720+00:00
Incoming items: 100

## Verdict

**ALERT**

| Test | Value | Threshold |
|---|---|---|
| PSI | 1.0487 | alert >= 0.25, watch >= 0.1 |
| Chi-square p | 0.0000 | reject at < 0.05 |

## Distributions

| Label | Baseline | Incoming | Delta |
|---|---|---|---|
| asthma | 0.043 | 0.150 | +0.107 |
| diabetes | 0.261 | 0.200 | -0.061 |
| hypertension | 0.217 | 0.550 | +0.333 |
| migraine | 0.478 | 0.100 | -0.378 |

PSI bands: < 0.10 no significant change; 0.10-0.25 watch; > 0.25 significant shift (standard industry bands).
