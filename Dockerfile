# ─────────────────────────────────────────────
# Stage 1 – builder: install Python dependencies
# ─────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS builder

WORKDIR /build

# Install build tools only in builder
RUN apt-get update && apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ─────────────────────────────────────────────
# Stage 2 – runtime: lean production image
# ─────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="Urdu Story AI" \
    org.opencontainers.image.description="FastAPI microservice for Urdu story generation" \
    org.opencontainers.image.source="https://github.com/$GITHUB_REPOSITORY"

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy only what the runtime actually needs
COPY main.py bpe_tokenizer.py ngram_model.py trigram_model.py preprocess.py ./
COPY bpe_vocab*.json ngram_n*.json trigram_model.json ./
COPY frontend/ ./frontend/

# Security: run as non-root user
RUN useradd --no-create-home --shell /bin/false appuser && \
    chown -R appuser:appuser /app
USER appuser

# Runtime configuration
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

EXPOSE 8000

# Health-check: poll /health every 30s; 3 failures = unhealthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
