# Production Dockerfile for Forex AI Super-Ensemble on Render
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-optimized PyTorch first (drastically reduces container image size)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy full application code
COPY . /app

# Expose web service port
EXPOSE 8080

# Launch server
CMD ["python", "dashboard_server.py", "--port", "8080"]
