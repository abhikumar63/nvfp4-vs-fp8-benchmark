FROM python:3.14-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and configs
COPY . /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Default command runs the local test config
CMD ["python", "scripts/run_benchmark.py", "--config", "config/test_local.yaml"]