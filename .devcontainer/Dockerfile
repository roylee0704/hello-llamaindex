FROM python:3.11-slim

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libffi-dev \
    libssl-dev \
    git \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, and wheel to ensure compatibility
RUN pip install --upgrade pip setuptools wheel

WORKDIR /workspace
