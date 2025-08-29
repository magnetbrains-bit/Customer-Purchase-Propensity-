# Smaller, newer base
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install Python deps first for better layer caching
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy app (includes model + data as you’ve structured it)
COPY . .

# Streamlit defaults
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

# Health check (optional)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD wget -qO- http://localhost:8501/_stcore/health || exit 1

# Support platforms that set $PORT (Render/Heroku) else default 8501
CMD ["bash","-lc","streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-8501}"]