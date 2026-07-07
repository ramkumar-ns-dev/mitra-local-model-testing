# ==========================================
# Stage 1: Build & Compile Dependencies
# ==========================================
FROM python:3.12-slim AS builder

WORKDIR /app

# Install compilation tools and git (needed for compiling C extensions and git dependencies)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment to isolate dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# OPTION A: For CPU-only deployment (Saves ~4.5 GB)
# Since default pip install on Linux pulls PyTorch CUDA wheels, we specify the CPU-only index.
# If you run on CPU/Mac locally, use this option:
RUN pip install --no-cache-dir torch==2.2.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cpu

# OPTION B: For GPU (CUDA) deployment (Uncomment below and comment out OPTION A if GPU is needed)
# RUN pip install --no-cache-dir torch==2.2.1 torchaudio==2.2.1

# Copy requirements file first to leverage Docker layer caching
COPY requirements.txt /app/

# Install the rest of the python dependencies
RUN pip install --no-cache-dir --default-timeout=300 -r requirements.txt

# ==========================================
# Stage 2: Clean Runtime Image
# ==========================================
FROM python:3.12-slim AS runner

WORKDIR /app

# Install only runtime system libraries (no build-essential, python3-dev, or git)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/model_cache

# Copy the backend source code directory to /app/backend
COPY . /app/backend

# Expose the port FastAPI runs on
EXPOSE 8000

# Start uvicorn server pointing to backend.main:app
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

