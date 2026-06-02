# Use Python 3.12 slim-buster as the base image for a lightweight footprint
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/model_cache

# Set working directory inside the container
WORKDIR /app

# Install system dependencies (git for pip, libsndfile/ffmpeg for audio, build-essential/python3-dev for compiling C extensions like psutil/IndicTransToolkit)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsndfile1 \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first to leverage Docker layer caching
COPY requirements.txt /app/

# Install python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the backend source code directory to /app/backend
COPY . /app/backend

# Expose the port FastAPI runs on
EXPOSE 8000

# Start uvicorn server pointing to backend.main:app
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
