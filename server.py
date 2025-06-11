import logging
import os
import glob
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import PlainTextResponse

# basic logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "ggml-small.bin")
BINARY     = "./main"
CHUNK_DIR  = "chunks"
CHUNK_DUR  = 35  # seconds
MAX_PARALLEL = 2  # run at most 2 whisper processes at once
WHISPER_THREADS = 1  # each whisper process will use 1 thread

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    # sanity-check model file
    if not os.path.isfile(MODEL_PATH):
        raise HTTPException(500, f"Model file not found at {MODEL_PATH}")
    logging.info(f"Loading model from {MODEL_PATH}")

    # save upload
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp_in:
        tmp_in.write(await audio.read())
        tmp_in_path = tmp_in.name

    tmp_wav = tmp_in_path + ".wav"
    os.makedirs(CHUNK_DIR, exist_ok=True)

    try:
        # convert to mono 16 kHz WAV
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_in_path, "-ar", "16000", "-ac", "1", tmp_wav],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        logging.info(f"Converted to WAV: {tmp_wav}")

        # split into 35s chunks, re-encoded to ensure valid PCM
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", tmp_wav,
                "-f", "segment", "-segment_time", str(CHUNK_DUR),
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                os.path.join(CHUNK_DIR, "chunk_%03d.wav")
            ],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # gather chunks
        chunk_paths = sorted(glob.glob(os.path.join(CHUNK_DIR, "*.wav")))
        if not chunk_paths:
            raise HTTPException(500, "No audio chunks were created.")
        for p in chunk_paths:
            size = os.path.getsize(p)
            logging.info(f"Chunk: {p} size={size} bytes")
            if size < 1024:
                logging.warning(f"Chunk {p} is very small; whisper may fail.")

        # worker that pins whisper to 1 thread and surfaces rc=-9
        def process_chunk(path: str) -> str:
            try:
                res = subprocess.run(
                    [
                        BINARY,
                        "-m", MODEL_PATH,
                        "-f", path,
                        "-otxt", "-of", "-",
                        "-nt", "-np",
                        "-t", str(WHISPER_THREADS),
                    ],
                    check=True, capture_output=True, text=True
                )
                return res.stdout
            except subprocess.CalledProcessError as e:
                rc = e.returncode
                if rc == -9:
                    hint = "Process was SIGKILLed (likely OOM). Try reducing parallelism or using more RAM."
                else:
                    hint = ""
                raise RuntimeError(
                    f"Chunk {os.path.basename(path)} failed (rc={rc}) {hint}\n"
                    f"--- stderr ---\n{e.stderr.strip()}"
                )

        # process in parallel but only 2 at a time
        transcripts = []
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
            futures = {pool.submit(process_chunk, p): p for p in chunk_paths}
            for fut in as_completed(futures):
                try:
                    transcripts.append(fut.result())
                except RuntimeError as err:
                    # immediately return the first chunk error
                    raise HTTPException(500, detail=str(err))

        # stitch and return
        return PlainTextResponse("\n".join(transcripts))

    finally:
        # cleanup
        for f in (tmp_in_path, tmp_wav):
            if os.path.exists(f):
                os.remove(f)
        shutil.rmtree(CHUNK_DIR, ignore_errors=True)
