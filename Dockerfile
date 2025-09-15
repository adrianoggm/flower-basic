# Dockerfile for Development Environment
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash flower
USER flower

# Set work directory
WORKDIR /app

# Copy dependency files
COPY --chown=flower:flower pyproject.toml requirements.txt ./

# Install Python dependencies
RUN pip install --user -r requirements.txt

# Copy source code
COPY --chown=flower:flower . .

# Install package in development mode
RUN pip install --user -e .

# Expose port for Flower server
EXPOSE 8080

# Default command
CMD ["python", "-m", "flower_basic"]