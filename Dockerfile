# Production annotation pipeline image.
# Build: docker build -t annotator-pipeline .
# Run:   docker run --rm annotator-pipeline python -m pipeline.annotate --help
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ANNOT_LOG_FORMAT=json

WORKDIR /app
COPY pipeline/requirements.txt /app/pipeline/requirements.txt
RUN pip install --no-cache-dir -r /app/pipeline/requirements.txt

COPY pipeline/ /app/pipeline/
COPY params.yaml /app/params.yaml

# The repo's research code (src/, data/) is mounted at runtime for the
# gemini backend and CSV extract; the image runs fully offline otherwise.
CMD ["python", "-m", "pipeline.annotate", "--help"]
