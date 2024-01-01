import whisper
from typing import BinaryIO, Union
from threading import Lock
import ffmpeg
import numpy as np

MODEL_NAME = "tiny"
SAMPLE_RATE = 16000


model = whisper.load_model(MODEL_NAME)
model_lock = Lock()

def transcribe(audio):
    with model_lock:
        result = model.transcribe(audio, fp16=False)
        return result['text'] 
    
def load_audio(file: BinaryIO, encode=True, sr: int = SAMPLE_RATE):
    if encode:
        try:
            with open(file, 'rb') as open_file:
                input_file = open_file.read()
            out, _ = (
                ffmpeg.input("pipe:", threads=0)
                .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
                .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True, input=input_file)
            )
        except ffmpeg.Error as e:
            raise RuntimeError(f"failed to load audio: {e.stderr.decode()}") from e
    else:
        with open(file, 'rb') as open_file:
            out = open_file.read()
    
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
    # return out
