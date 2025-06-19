import logging
import os
import glob
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import PlainTextResponse

logging.basicConfig(level=logging.INFO)
app = FastAPI()

MODEL_PATH     = os.path.join(os.path.dirname(__file__), "models", "ggml-small.bin")
BINARY         = "./main"
CHUNK_DIR      = "chunks"
CHUNK_DUR      = 35     # seconds
MAX_PARALLEL   = 4      # only for long audio
WHISPER_THREADS = 2     # whisper.cpp -t 1

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    # ensure model exists
    if not os.path.isfile(MODEL_PATH):
        raise HTTPException(500, f"Model not found at {MODEL_PATH}")
    logging.info(f"Using model: {MODEL_PATH}")

    # 1) save incoming upload
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp:
        tmp.write(await audio.read())
        tmp_in = tmp.name

    tmp_wav = tmp_in + ".wav"
    os.makedirs(CHUNK_DIR, exist_ok=True)

    try:
        # 2) convert to standard mono WAV
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_in, "-ar", "16000", "-ac", "1", tmp_wav],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # 3) probe duration
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
        logging.info(f"Audio duration: {duration:.2f}s")

        # 4) if short, run whisper once and return
        if duration <= CHUNK_DUR:
            logging.info("Short audio — skipping chunking.")
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
                return PlainTextResponse(res.stdout)
            except subprocess.CalledProcessError as e:
                raise HTTPException(
                    500,
                    detail=f"Whisper failed (rc={e.returncode}): {e.stderr.strip()}"
                )

        # 5) long audio — split into re-encoded 35s chunks
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", tmp_wav,
                "-f", "segment", "-segment_time", str(CHUNK_DUR),
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                os.path.join(CHUNK_DIR, "chunk_%03d.wav")
            ],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # 6) collect chunks
        chunks = sorted(glob.glob(os.path.join(CHUNK_DIR, "*.wav")))
        if not chunks:
            raise HTTPException(500, "No audio chunks were created.")
        for p in chunks:
            size = os.path.getsize(p)
            logging.info(f"Chunk {os.path.basename(p)} → {size} bytes")

        # 7) worker that pins to 1 thread and surfaces OOM kills
        def process_chunk(path: str) -> str:
            try:
                out = subprocess.run(
                    [
                        BINARY, "-m", MODEL_PATH, "-f", path,
                        "-otxt", "-of", "-", "-nt", "-np",
                        "-t", str(WHISPER_THREADS)
                    ],
                    check=True, capture_output=True, text=True
                )
                return out.stdout
            except subprocess.CalledProcessError as e:
                hint = " (SIGKILL/oom)" if e.returncode == -9 else ""
                raise RuntimeError(
                    f"{os.path.basename(path)} failed rc={e.returncode}{hint}\n"
                    f"{e.stderr.strip()}"
                )

        # 8) run in parallel but limited workers
        results = {}
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
            futures = {pool.submit(process_chunk, p): p for p in chunks}
            for fut in as_completed(futures):
                p = futures[fut]
                try:
                    results[p] = fut.result()
                except RuntimeError as e:
                    raise HTTPException(500, detail=str(e))

        # 9) stitch in order and return
        ordered = [results[p] for p in chunks]
        return PlainTextResponse("\n".join(ordered))

    finally:
        # cleanup
        for f in (tmp_in, tmp_wav):
            if os.path.exists(f):
                os.remove(f)
        shutil.rmtree(CHUNK_DIR, ignore_errors=True)
