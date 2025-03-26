''' infer.py for runpod worker '''

import os
import predict
import base64
import tempfile

import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils import download_files_from_urls, rp_cleanup

from rp_schema import INPUT_VALIDATIONS

MODEL = predict.Predictor()
MODEL.setup()

def base64_to_tempfile(base64_file: str) -> str:
    '''
    Convert base64 file to tempfile.

    Parameters:
    base64_file (str): Base64 file

    Returns:
    str: Path to tempfile
    '''
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file.write(base64.b64decode(base64_file))

    return temp_file.name



def run(job):

    job_input = job['input']

    input_validation = validate(job_input, INPUT_VALIDATIONS)

    if 'errors' in input_validation:
        return {"error": input_validation['errors']}
    job_input = input_validation['validated_input']

    if not job_input.get('audio', False) and not job_input.get('audio_base64', False):
        return {'error': 'Must provide either audio or audio_base64'}

    if job_input.get('audio', False) and job_input.get('audio_base64', False):
        return {'error': 'Must provide either audio or audio_base64, not both'}

    if job_input.get('audio', False):
        audio_input = download_files_from_urls(job['id'], [job_input['audio']])[0]

    if job_input.get('audio_base64', False):
        audio_input = base64_to_tempfile(job_input['audio_base64'])

    whisper_results = MODEL.predict(
        audio=job_input["audio"],
        model_name=job_input.get("model", 'base'),
        transcription=job_input.get('transcription', 'plain_text'),
        translate=job_input.get('translate', False),
        language=job_input.get('language', None),
        temperature=job_input["temperature"],
        best_of=job_input.get("best_of", 5),
        beam_size=job_input.get("beam_size", 5),
        patience=job_input["patience"],
        length_penalty=job_input["length_penalty"],
        suppress_tokens=job_input.get("suppress_tokens", "-1"),
        initial_prompt=job_input.get('initial_prompt', None),
        condition_on_previous_text=job_input.get('condition_on_previous_text', True),
        temperature_increment_on_fallback=job_input["temperature_increment_on_fallback"],
        compression_ratio_threshold=job_input["compression_ratio_threshold"],
        logprob_threshold=job_input["logprob_threshold"],
        no_speech_threshold=job_input["no_speech_threshold"],
    )

    rp_cleanup.clean(['input_objects'])

    return whisper_results


runpod.serverless.start({"handler": run})
