import logging
import os
import glob
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import PlainTextResponse

logging.basicConfig(level=logging.INFO)
app = FastAPI()

MODEL_PATH       = os.path.join(os.path.dirname(__file__), "models", "ggml-small-q5_1.bin")
BINARY           = "./main"
CHUNK_DIR        = "chunks"
CHUNK_DUR        = 35     # seconds
MAX_PARALLEL     = 2      # concurrent chunk workers
WHISPER_THREADS  = os.cpu_count() or 2

@app.on_event("startup")
def startup_log():
    logging.info(f"🧠 Model in use: {MODEL_PATH}")
    logging.info(f"⚙️ Whisper threads: {WHISPER_THREADS}")
    logging.info(f"🚀 Max parallel chunks: {MAX_PARALLEL}")
    logging.info(f"📦 Detected CPU cores: {os.cpu_count()}")

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    if not os.path.isfile(MODEL_PATH):
        raise HTTPException(500, f"Model not found at {MODEL_PATH}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp:
        tmp.write(await audio.read())
        tmp_in = tmp.name

    tmp_wav = tmp_in + ".wav"
    os.makedirs(CHUNK_DIR, exist_ok=True)

    try:
        # Convert to WAV
        t0 = time.perf_counter()
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_in, "-ar", "16000", "-ac", "1", tmp_wav],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        logging.info(f"🎧 ffmpeg convert time: {time.perf_counter() - t0:.2f}s")

        # Probe duration
        ff = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                tmp_wav
            ],
            capture_output=True, text=True
        )
        try:
            duration = float(ff.stdout.strip())
        except:
            duration = 0.0
        logging.info(f"🎙 Audio duration: {duration:.2f}s")

        # Short audio (no chunking)
        if duration <= CHUNK_DUR:
            logging.info("✅ Short audio — skipping chunking.")
            t1 = time.perf_counter()
            try:
                res = subprocess.run(
                    [
                        BINARY, "-m", MODEL_PATH,
                        "-f", tmp_wav,
                        "-otxt", "-of", "-",
                        "-nt", "-np",
                        "-t", str(WHISPER_THREADS)
                    ],
                    check=True, capture_output=True, text=True
                )
                logging.info(f"⏱ whisper.cpp runtime: {time.perf_counter() - t1:.2f}s")
                return PlainTextResponse(res.stdout)
            except subprocess.CalledProcessError as e:
                raise HTTPException(
                    500,
                    detail=f"Whisper failed (rc={e.returncode}): {e.stderr.strip()}"
                )

        # Long audio — chunk
        t2 = time.perf_counter()
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", tmp_wav,
                "-f", "segment", "-segment_time", str(CHUNK_DUR),
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                os.path.join(CHUNK_DIR, "chunk_%03d.wav")
            ],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        logging.info(f"🔪 Chunking time: {time.perf_counter() - t2:.2f}s")

        # Collect chunks
        chunks = sorted(glob.glob(os.path.join(CHUNK_DIR, "*.wav")))
        if not chunks:
            raise HTTPException(500, "No audio chunks were created.")
        for p in chunks:
            logging.info(f"📦 Chunk {os.path.basename(p)} → {os.path.getsize(p)} bytes")

        # Transcribe each chunk
        def process_chunk(path: str) -> str:
            t_chunk = time.perf_counter()
            try:
                out = subprocess.run(
                    [
                        BINARY, "-m", MODEL_PATH, "-f", path,
                        "-otxt", "-of", "-", "-nt", "-np",
                        "-t", str(WHISPER_THREADS)
                    ],
                    check=True, capture_output=True, text=True
                )
                logging.info(f"🧩 {os.path.basename(path)} done in {time.perf_counter() - t_chunk:.2f}s")
                return out.stdout
            except subprocess.CalledProcessError as e:
                hint = " (SIGKILL/oom)" if e.returncode == -9 else ""
                raise RuntimeError(
                    f"{os.path.basename(path)} failed rc={e.returncode}{hint}\n"
                    f"{e.stderr.strip()}"
                )

        results = {}
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
            futures = {pool.submit(process_chunk, p): p for p in chunks}
            for fut in as_completed(futures):
                p = futures[fut]
                try:
                    results[p] = fut.result()
                except RuntimeError as e:
                    raise HTTPException(500, detail=str(e))

        ordered = [results[p] for p in chunks]
        return PlainTextResponse("\n".join(ordered))

    finally:
        for f in (tmp_in, tmp_wav):
            if os.path.exists(f):
                os.remove(f)
        shutil.rmtree(CHUNK_DIR, ignore_errors=True)
