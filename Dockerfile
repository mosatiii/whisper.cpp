FROM python:3.11-slim

# install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ffmpeg wget git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# build whisper.cpp binary using CMake
RUN cmake -B build
RUN cmake --build build --config Release -j$(nproc)

# create a symlink to whisper-cli
RUN ln -s build/bin/whisper-cli /app/main

# download model
RUN mkdir -p models && \
    wget -O models/ggml-medium.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin

# install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
