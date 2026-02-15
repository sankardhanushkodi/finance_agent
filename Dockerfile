# ──────────────────────────────────────────────────────────────────────────────
# Finance Multi-Agent System — Dockerfile
# ──────────────────────────────────────────────────────────────────────────────
# Targets Hugging Face Spaces (Docker SDK).  HF Spaces requirements:
#   • App must listen on port 7860
#   • Container runs as UID 1000 (non-root)
#   • Secrets (API keys) are set in Space Settings → Repository Secrets,
#     NOT baked into the image.
#
# Local build & run:
#   docker build -t finance-agent .
#   docker run -p 7860:7860 \
#       -e ANTHROPIC_API_KEY=sk-ant-... \
#       -e LLM_PROVIDER=anthropic \
#       finance-agent
#
# Then open http://localhost:7860
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Create non-root user matching HF Spaces UID ───────────────────────────────
RUN useradd -m -u 1000 appuser

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies (own separate layer for cache efficiency) ─────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Application code (exclude secrets/venv via .dockerignore) ─────────────────
COPY --chown=appuser:appuser . .

# ── Switch to non-root user ───────────────────────────────────────────────────
USER appuser

# ── Streamlit / HF Spaces configuration ──────────────────────────────────────
ENV STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_THEME_BASE=dark

# LLM defaults – override via HF Spaces Secrets or docker run -e
ENV LLM_PROVIDER=anthropic \
    LLM_MODEL=claude-haiku-4-5-20251001

EXPOSE 7860

CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
