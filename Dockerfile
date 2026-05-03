# CUDA-accelerated image for stream-translator-gpt
#
# Build:
#   docker build -t stream-translator-gpt .
#
# Run (CLI):
#   docker run --rm --gpus all stream-translator-gpt {URL} --model large --language ja
#
# Run (WebUI):
#   docker run --rm --gpus all -p 7860:7860 stream-translator-gpt-webui
#
# Requires NVIDIA Container Toolkit on the host:
#   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System dependencies:
#   ffmpeg         - audio format conversion
#   portaudio19-dev, python3-dev, build-essential
#                  - required to compile pyaudio for device audio capture
#   git            - needed by some pip packages during install
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    ffmpeg \
    portaudio19-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch with CUDA 12.4 support BEFORE the application.
# This prevents openai-whisper from pulling in CPU-only torch.
RUN pip3 install --no-cache-dir \
    torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# Copy project files
COPY . .

# Install the application with all dependencies, including WebUI
RUN pip3 install --no-cache-dir ".[webui]"

# Install pyaudio for device audio capture on Linux (optional at
# runtime, but available if the container is used with --device)
RUN pip3 install --no-cache-dir pyaudio

# Gradio WebUI default port
EXPOSE 7860

ENTRYPOINT ["stream-translator-gpt"]
