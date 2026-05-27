# FairLens — Production Dockerfile for Google Cloud Run
# Build: docker build -t fairlens .
# Run locally: docker run -p 8080:8080 fairlens

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and artifacts
COPY app.py .
COPY artifacts/ ./artifacts/
COPY .streamlit/ ./.streamlit/

# Cloud Run passes PORT env variable; Streamlit must listen on it
ENV PORT=8080
EXPOSE 8080

# Health check for Cloud Run
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s \
  CMD curl -f http://localhost:8080/_stcore/health || exit 1

# Run Streamlit on Cloud Run's PORT
CMD streamlit run app.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false
