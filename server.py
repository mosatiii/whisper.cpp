import logging
import os
import glob
import shutil
import subprocess
import tempfile
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, PlainTextResponse

logging.basicConfig(level=logging.INFO)
app = FastAPI()

MODEL_PATH     = os.path.join(os.path.dirname(__file__), "models", "ggml-small.bin")
BINARY         = "./main"
CHUNK_DIR      = "chunks"
CHUNK_DUR      = 35
MAX_PARALLEL   = 4
WHISPER_THREADS = 2

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    task_id = str(uuid.uuid4())
    tmp_audio_path = os.path.join(RESULTS_DIR, f"{task_id}.upload")

    with open(tmp_audio_path, "wb") as f:
        f.write(await audio.read())

    background_tasks.add_task(process_transcription, tmp_audio_path, task_id)
    return JSONResponse({"task_id": task_id, "status": "processing"}, status_code=202)


@app.get("/result/{task_id}")
async def get_result(task_id: str):
    result_path = os.path.join(RESULTS_DIR, f"{task_id}.txt")
    if os.path.exists(result_path):
        with open(result_path, "r") as f:
            return PlainTextResponse(f.read())
    return JSONResponse({"status": "processing"}, status_code=202)


def process_transcription(tmp_in: str, task_id: str):
    tmp_wav = tmp_in + ".wav"
    os.makedirs(CHUNK_DIR, exist_ok=True)

    try:
        # Convert to mono WAV
        subprocess.run(["ffmpeg", "-y", "-i", tmp_in, "-ar", "16000", "-ac", "1", tmp_wav],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Check duration
        duration = float(subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", tmp_wav],
            capture_output=True, text=True).stdout.strip())

        if duration <= CHUNK_DUR:
            result = subprocess.run([BINARY, "-m", MODEL_PATH, "-f", tmp_wav, "-otxt", "-of", "-", "-nt", "-np", "-t", str(WHISPER_THREADS)],
                                    check=True, capture_output=True, text=True).stdout
            with open(os.path.join(RESULTS_DIR, f"{task_id}.txt"), "w") as f:
                f.write(result)
            return

        # Chunking
        subprocess.run(["ffmpeg", "-y", "-i", tmp_wav, "-f", "segment", "-segment_time", str(CHUNK_DUR),
                        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", os.path.join(CHUNK_DIR, "chunk_%03d.wav")],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        chunks = sorted(glob.glob(os.path.join(CHUNK_DIR, "*.wav")))

        from concurrent.futures import ThreadPoolExecutor, as_completed
        def process_chunk(path):
            return subprocess.run([BINARY, "-m", MODEL_PATH, "-f", path, "-otxt", "-of", "-", "-nt", "-np", "-t", str(WHISPER_THREADS)],
                                  check=True, capture_output=True, text=True).stdout

        results = []
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
            futures = [pool.submit(process_chunk, c) for c in chunks]
            for fut in as_completed(futures):
                results.append(fut.result())

        with open(os.path.join(RESULTS_DIR, f"{task_id}.txt"), "w") as f:
            f.write("\n".join(results))

    except Exception as e:
        logging.error(f"Transcription failed for {task_id}: {str(e)}")

    finally:
        for f in [tmp_in, tmp_wav]:
            if os.path.exists(f):
                os.remove(f)
        shutil.rmtree(CHUNK_DIR, ignore_errors=True)
