FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ffmpeg wget git pkg-config && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app
COPY . /app

# Build whisper.cpp with full Release optimizations
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --config Release -j$(nproc)

# Symlink to final binary name
RUN ln -s build/bin/whisper-cli /app/main

# Download quantized model (fast + memory-efficient)
RUN mkdir -p models && \
    wget -O models/ggml-small-q5_1.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small-q5_1.bin

# Install Python requirements
RUN pip install --no-cache-dir -r requirements.txt

# Expose port and run API
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
