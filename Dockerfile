# Base image with PyTorch and CUDA
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    redis-server \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    MODEL_CACHE_DIR=/runpod-volume/models \
    OUTPUT_DIR=/runpod-volume/outputs \
    REDIS_HOST=localhost \
    REDIS_PORT=6379

# Create necessary directories
RUN mkdir -p ${MODEL_CACHE_DIR} ${OUTPUT_DIR} /app

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the application code
COPY . .

# Download models during build (uncomment if you want to bake models into the image)
# RUN python -c "from src.models.model_loader import download_model; download_model('tencent/hunyuan3d-2.1', '/runpod-volume/models')"

# Expose ports
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 CMD curl -f http://localhost:8000/health || exit 1

# Start script
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Default command
CMD ["/start.sh"]
