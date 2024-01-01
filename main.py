from fastapi import FastAPI
from handler import handle_transcribe
from pydantic import BaseModel

class TranscriptionRequest(BaseModel):
    bucket: str
    key: str


app = FastAPI()

@app.post('/transcribe')
async def transcribe(body: TranscriptionRequest):
    response = handle_transcribe(body.bucket, body.key)
    return response