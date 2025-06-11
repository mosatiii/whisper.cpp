from fastapi import FastAPI, File, UploadFile
from fastapi.responses import PlainTextResponse
import subprocess
import tempfile
import os

app = FastAPI()

MODEL_PATH = "models/ggml-medium.bin"
BINARY = "./main"

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_in:
        data = await audio.read()
        tmp_in.write(data)
        tmp_in_path = tmp_in.name
    tmp_wav = tmp_in_path + ".wav"
    try:
        subprocess.run([
            "ffmpeg","-y","-i",tmp_in_path,"-ar","16000","-ac","1",tmp_wav
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        result = subprocess.run([
            BINARY,
            "-m", MODEL_PATH,
            "-f", tmp_wav,
            "-otxt",
            "-of", "-",
            "-nt",
            "-np"
        ], check=True, capture_output=True, text=True)
        return PlainTextResponse(result.stdout)
    finally:
        if os.path.exists(tmp_in_path):
            os.remove(tmp_in_path)
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)

