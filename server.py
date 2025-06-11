import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import PlainTextResponse
import subprocess
import tempfile
import os
import shutil
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

# set up basic logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "ggml-small.bin")
BINARY     = "./main"
CHUNK_DIR  = "chunks"
CHUNK_DUR  = 35  # seconds

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    # 0) sanity-check model
    if not os.path.isfile(MODEL_PATH):
        raise HTTPException(500, f"Model file not found at {MODEL_PATH}")
    logging.info(f"Using model at {MODEL_PATH}")

    # 1) save upload
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp_in:
        tmp_in.write(await audio.read())
        tmp_in_path = tmp_in.name

    tmp_wav = tmp_in_path + ".wav"
    os.makedirs(CHUNK_DIR, exist_ok=True)

    try:
        # 2) convert to 16k mono WAV
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_in_path, "-ar", "16000", "-ac", "1", tmp_wav],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        logging.info(f"Converted upload to WAV: {tmp_wav}")

        # 3) split into 35s chunks *with re-encoding* to ensure valid PCM
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", tmp_wav,
                "-f", "segment", "-segment_time", str(CHUNK_DUR),
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                os.path.join(CHUNK_DIR, "chunk_%03d.wav")
            ],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # 4) list & log chunk files
        chunk_paths = sorted(glob.glob(os.path.join(CHUNK_DIR, "*.wav")))
        if not chunk_paths:
            raise HTTPException(500, "No audio chunks were created.")
        for p in chunk_paths:
            size = os.path.getsize(p)
            logging.info(f"Chunk: {p} size={size} bytes")
            if size < 1024:
                logging.warning(f"Chunk {p} is tiny—whisper may fail on it")

        # 5) worker that captures return code, stdout & stderr
        def process_chunk(path: str) -> str:
            try:
                res = subprocess.run(
                    [BINARY, "-m", MODEL_PATH, "-f", path, "-otxt", "-of", "-", "-nt", "-np"],
                    check=True, capture_output=True, text=True
                )
                return res.stdout
            except subprocess.CalledProcessError as e:
                # include rc, stdout and stderr in the error
                raise RuntimeError(
                    f"Chunk {os.path.basename(path)} failed (rc={e.returncode})\n"
                    f"--- stdout ---\n{e.stdout.strip()}\n"
                    f"--- stderr ---\n{e.stderr.strip()}"
                )

        # 6) parallel processing with as_completed
        transcripts = []
        with ThreadPoolExecutor() as pool:
            futures = {pool.submit(process_chunk, p): p for p in chunk_paths}
            for fut in as_completed(futures):
                try:
                    transcripts.append(fut.result())
                except RuntimeError as err:
                    # immediate 500 with full debug info
                    raise HTTPException(500, detail=str(err))

        # 7) stitch and return
        return PlainTextResponse("\n".join(transcripts))

    finally:
        # cleanup everything
        for f in (tmp_in_path, tmp_wav):
            if os.path.exists(f):
                os.remove(f)
        shutil.rmtree(CHUNK_DIR, ignore_errors=True)
