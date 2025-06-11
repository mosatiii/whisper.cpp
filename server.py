from fastapi import FastAPI, File, UploadFile
from fastapi.responses import PlainTextResponse
import subprocess
import tempfile
import os
import shutil
import glob
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()

MODEL_PATH   = "models/ggml-small.bin"
BINARY       = "./main"
CHUNK_DIR    = "chunks"
CHUNK_DUR    = 35  # seconds

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    # 1. Store incoming upload
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp_in:
        tmp_in.write(await audio.read())
        tmp_in_path = tmp_in.name

    # 2. Convert to mono WAV @16 kHz
    tmp_wav = tmp_in_path + ".wav"
    os.makedirs(CHUNK_DIR, exist_ok=True)
    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-i", tmp_in_path,
            "-ar", "16000", "-ac", "1",
            tmp_wav
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # 3. Split into 35 s segments in CHUNK_DIR
        subprocess.run([
            "ffmpeg", "-y",
            "-i", tmp_wav,
            "-f", "segment",
            "-segment_time", str(CHUNK_DUR),
            "-c", "copy",
            os.path.join(CHUNK_DIR, "chunk_%03d.wav")
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # 4. Define worker to run each chunk through whisper.cpp
        def process_chunk(path: str) -> str:
            res = subprocess.run([
                BINARY,
                "-m", MODEL_PATH,
                "-f", path,
                "-otxt", "-of", "-",
                "-nt", "-np"
            ], check=True, capture_output=True, text=True)
            return res.stdout

        # 5. Dispatch them in parallel threads
        transcripts = []
        chunk_paths = sorted(glob.glob(os.path.join(CHUNK_DIR, "*.wav")))
        with ThreadPoolExecutor() as pool:
            for part in pool.map(process_chunk, chunk_paths):
                transcripts.append(part)

        # 6. Return the full combined transcript
        return PlainTextResponse("\n".join(transcripts))

    finally:
        # 7. Cleanup all temp files & chunk directory
        if os.path.exists(tmp_in_path): os.remove(tmp_in_path)
        if os.path.exists(tmp_wav):      os.remove(tmp_wav)
        shutil.rmtree(CHUNK_DIR, ignore_errors=True)
