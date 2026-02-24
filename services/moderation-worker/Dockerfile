FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# LAYER 1: Heavy ML dependencies (cached)
# These rarely change, so Docker can cache this layer
# ============================================
COPY requirements-base.txt .
RUN pip install --no-cache-dir -r requirements-base.txt

# ============================================
# LAYER 2: Light app dependencies
# These may change occasionally
# ============================================
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

# ============================================
# LAYER 3: Application code (changes frequently)
# Only this layer needs to rebuild when code changes
# ============================================
COPY . .

# Create models directory
RUN mkdir -p /app/models

# Run worker
CMD ["python", "worker.py"]
