FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Download model checkpoints
RUN huggingface-cli download ByteDance/LatentSync-1.6 whisper/tiny.pt --local-dir checkpoints && \
    huggingface-cli download ByteDance/LatentSync-1.6 latentsync_unet.pt --local-dir checkpoints

# Environment variables (can be overridden at runtime)
ENV WEBSERVER_URL=""
ENV API_AUTH_TOKEN=""
ENV PYTORCH_ENABLE_MPS_FALLBACK=1

# Run the service
CMD ["python", "latentsync_service.py"]