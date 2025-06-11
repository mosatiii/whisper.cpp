from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import PlainTextResponse
import subprocess
import tempfile
import os
import shutil
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

app = FastAPI()

MODEL_PATH = "models/ggml-small.bin"
BINARY     = "./main"
CHUNK_DIR  = "chunks"
CHUNK_DUR  = 35  # seconds

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    # 1) Save incoming upload to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp_in:
        tmp_in.write(await audio.read())
        tmp_in_path = tmp_in.name

    # 2) Define paths and ensure chunk dir exists
    tmp_wav = tmp_in_path + ".wav"
    os.makedirs(CHUNK_DIR, exist_ok=True)

    try:
        # 3) Convert to mono 16 kHz WAV
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_in_path, "-ar", "16000", "-ac", "1", tmp_wav],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # 4) Split into 35s chunks
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", tmp_wav,
                "-f", "segment", "-segment_time", str(CHUNK_DUR),
                "-c", "copy",
                os.path.join(CHUNK_DIR, "chunk_%03d.wav")
            ],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # 5) Gather chunk file paths
        chunk_paths = sorted(glob.glob(os.path.join(CHUNK_DIR, "*.wav")))
        if not chunk_paths:
            raise HTTPException(500, "No audio chunks were created.")

        # 6) Define a worker that captures stderr on failure
        def process_chunk(path: str) -> str:
            try:
                res = subprocess.run(
                    [
                        BINARY,
                        "-m", MODEL_PATH,
                        "-f", path,
                        "-otxt", "-of", "-",
                        "-nt", "-np"
                    ],
                    check=True, capture_output=True, text=True
                )
                return res.stdout
            except subprocess.CalledProcessError as e:
                err = e.stderr.strip()
                raise RuntimeError(f"Chunk {os.path.basename(path)} failed:\n{err}")

        # 7) Process all chunks in parallel threads
        transcripts = []
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(process_chunk, p): p for p in chunk_paths}
            for future in as_completed(futures):
                try:
                    transcripts.append(future.result())
                except RuntimeError as e:
                    # Stop on first chunk error, return its stderr
                    raise HTTPException(500, detail=str(e))

        # 8) Return combined transcript
        return PlainTextResponse("\n".join(transcripts))

    finally:
        # 9) Cleanup temp files and chunk directory
        if os.path.exists(tmp_in_path):
            os.remove(tmp_in_path)
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)
        shutil.rmtree(CHUNK_DIR, ignore_errors=True)
