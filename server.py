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

MODEL_PATH    = os.path.join(os.path.dirname(__file__), "models", "ggml-small.bin")
BINARY        = "./main"
CHUNK_DIR     = "chunks"
CHUNK_DUR     = 35     # seconds
MAX_PARALLEL  = 2      # limit simultaneous whisper processes
WHISPER_THREADS = 1    # whisper.cpp -t 1

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    # sanity-check model
    if not os.path.isfile(MODEL_PATH):
        raise HTTPException(500, f"Model not found at {MODEL_PATH}")
    logging.info(f"Using model: {MODEL_PATH}")

    # save upload
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp:
        tmp.write(await audio.read())
        inp = tmp.name

    wav = inp + ".wav"
    os.makedirs(CHUNK_DIR, exist_ok=True)

    try:
        # convert to 16 kHz mono WAV
        subprocess.run(
            ["ffmpeg", "-y", "-i", inp, "-ar", "16000", "-ac", "1", wav],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # split into 35 s re-encoded chunks
        subprocess.run([
            "ffmpeg", "-y", "-i", wav,
            "-f", "segment", "-segment_time", str(CHUNK_DUR),
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            os.path.join(CHUNK_DIR, "chunk_%03d.wav")
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # gather chunk files
        chunks = sorted(glob.glob(os.path.join(CHUNK_DIR, "*.wav")))
        if not chunks:
            raise HTTPException(500, "No audio chunks created.")
        for p in chunks:
            size = os.path.getsize(p)
            logging.info(f"{p} → {size} bytes")
            if size < 1024:
                logging.warning(f"{p} is tiny; whisper may choke on it.")

        # worker invoking whisper.cpp
        def process_chunk(path: str) -> str:
            try:
                out = subprocess.run([
                    BINARY, "-m", MODEL_PATH, "-f", path,
                    "-otxt", "-of", "-", "-nt", "-np",
                    "-t", str(WHISPER_THREADS)
                ], check=True, capture_output=True, text=True)
                return out.stdout
            except subprocess.CalledProcessError as e:
                hint = " (SIGKILL/oom?)" if e.returncode == -9 else ""
                raise RuntimeError(
                    f"{os.path.basename(path)} failed rc={e.returncode}{hint}\n"
                    f"--- stderr ---\n{e.stderr.strip()}"
                )

        # run in parallel, collect into dict
        results = {}
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
            futures = {pool.submit(process_chunk, p): p for p in chunks}
            for fut in as_completed(futures):
                p = futures[fut]
                try:
                    results[p] = fut.result()
                except RuntimeError as e:
                    raise HTTPException(500, detail=str(e))

        # stitch in chunk order
        ordered = [results[p] for p in chunks]
        return PlainTextResponse("\n".join(ordered))

    finally:
        # cleanup
        for f in (inp, wav):
            if os.path.exists(f):
                os.remove(f)
        shutil.rmtree(CHUNK_DIR, ignore_errors=True)
