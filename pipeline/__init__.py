"""Production annotation pipeline for the llm-annotator-reliability research.

Batch annotation infrastructure: extract -> chunk -> dedup -> annotate ->
agreement metrics -> versioned results, with experiment tracking, drift
monitoring, and AI-assisted adversarial analysis.
"""

__version__ = "0.1.0"

# Data-contract version. Stamped into every batch manifest and baseline file.
# Bump on any schema change and note the migration in the README changelog.
CONTRACT_VERSION = "1.0"
