import boto3
import os
import tempfile
from transcribe import load_audio, transcribe

FILE_BUCKET = 'penman_transcription'


def read_content_from_s3(bucket, key, temp_file):
    s3 = boto3.client('s3')
    s3.download_file(bucket, key, temp_file)

def handle_transcribe(bucket, key):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_file = os.path.join(tmpdir, key)

        read_content_from_s3(bucket, key, tmp_file)

        response = transcribe(load_audio(tmp_file))

        result = {
            'transcription': response
        }

        return result
