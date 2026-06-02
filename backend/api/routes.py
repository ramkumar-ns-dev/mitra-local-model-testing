import base64
import requests
import time
from fastapi import APIRouter, Request, HTTPException, status
from pydantic import BaseModel, Field
from backend.services.audio_processor import AudioProcessor
from backend.services.model_loader import model_loader
from backend.utils.logging_config import get_logger

logger = get_logger()
router = APIRouter()

# Translation Request Schema
class TranslateRequest(BaseModel):
    text: str = Field(..., description="The text to translate")
    source_lang: str = Field(..., description="The source language tag, e.g. eng_Latn")
    target_lang: str = Field(..., description="The target language tag, e.g. hin_Deva")

@router.post("/transcribe")
async def transcribe(request: Request):
    """
    Speech-to-Text Endpoint.
    Accepts audio data via:
    1. Multipart form-data file upload (field name: 'audio')
    2. JSON payload with 'audio_url'
    3. JSON payload with 'audio_base64' (optional filename inside payload)
    """
    start_time = time.time()
    content_type = request.headers.get("content-type", "")
    
    file_bytes = None
    filename = "audio.wav"
    file_size = 0
    
    try:
        # Parse inputs based on Request Content-Type
        if "multipart/form-data" in content_type:
            logger.info("Parsing multipart form-data upload for transcription")
            form = await request.form()
            upload_file = form.get("audio")
            if not upload_file:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, 
                    detail="Missing 'audio' field in multipart form-data upload"
                )
            filename = upload_file.filename
            file_bytes = await upload_file.read()
            file_size = len(file_bytes)
            
        elif "application/json" in content_type:
            logger.info("Parsing JSON payload for transcription")
            try:
                body = await request.json()
            except Exception:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, 
                    detail="Invalid JSON format"
                )
            
            audio_url = body.get("audio_url")
            audio_base64 = body.get("audio_base64")
            
            if not audio_url and not audio_base64:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="JSON payload must contain either 'audio_url' or 'audio_base64'"
                )
            
            if audio_url:
                logger.info(f"Downloading audio from URL: {audio_url}")
                try:
                    # Download audio file with a 15-second timeout
                    response = requests.get(audio_url, timeout=15.0)
                    response.raise_for_status()
                    file_bytes = response.content
                    file_size = len(file_bytes)
                    
                    # Deduce filename from URL or default
                    url_path = audio_url.split("?")[0]
                    filename = url_path.split("/")[-1] or "audio.wav"
                except requests.exceptions.RequestException as e:
                    logger.error(f"Failed to fetch audio from URL: {e}")
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Failed to download audio from URL: {str(e)}"
                    )
            else:
                logger.info("Decoding base64 audio string")
                try:
                    # Strip standard base64 data URL headers if present
                    if "," in audio_base64:
                        audio_base64 = audio_base64.split(",")[1]
                    file_bytes = base64.b64decode(audio_base64)
                    file_size = len(file_bytes)
                    filename = body.get("filename", "audio.wav")
                except Exception as e:
                    logger.error(f"Failed to decode base64 audio: {e}")
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Failed to decode base64 audio. Ensure the base64 string is well-formed."
                    )
        else:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Unsupported Content-Type header. Use multipart/form-data or application/json."
            )
            
        # 2. Validation
        AudioProcessor.validate_file_metadata(filename, content_type=None, file_size=file_size)
        
        # 3. Audio Preprocessing
        waveform, duration = AudioProcessor.process_audio(file_bytes, filename)
        
        # 4. Inference
        inference_start = time.time()
        inference_result = model_loader.transcribe(waveform)
        inference_time_ms = int((time.time() - inference_start) * 1000)
        
        total_time_ms = int((time.time() - start_time) * 1000)
        logger.info(
            f"Transcription completed: duration={duration:.2f}s, size={file_size} bytes, "
            f"inference_time={inference_time_ms}ms, total_time={total_time_ms}ms, engine={inference_result['engine']}"
        )
        
        response_payload = {
            "success": True,
            "text": inference_result["text"]
        }
        
        # Include confidence score if provided by model
        if inference_result.get("confidence") is not None:
            response_payload["confidence"] = inference_result["confidence"]
            
        return response_payload

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error during transcription: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal transcription failure: {str(e)}"
        )

@router.post("/translate")
async def translate(body: TranslateRequest):
    """
    Translation Endpoint.
    Translates text from source_lang to target_lang using NLLB-200.
    """
    start_time = time.time()
    logger.info(f"Translation request: src={body.source_lang}, tgt={body.target_lang}, text_len={len(body.text)}")
    
    try:
        translated_text = model_loader.translate(
            text=body.text,
            source_lang=body.source_lang,
            target_lang=body.target_lang
        )
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        logger.info(f"Translation completed in {processing_time_ms}ms")
        
        return {
            "translation": translated_text
        }
    except Exception as e:
        logger.exception(f"Translation failure: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Translation failure: {str(e)}"
        )

@router.get("/health")
async def health():
    """
    Health Check Endpoint.
    Returns status indicators for ASR and translation services.
    """
    status_info = model_loader.get_status()
    
    # Ready if loaded successfully or in mock mode
    asr_ready = status_info["asr_status"] in ("ready", "ready_mock")
    translation_ready = status_info["translation_status"] in ("ready", "ready_mock")
    
    is_healthy = asr_ready and translation_ready
    
    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "model_loaded": is_healthy,
        "details": status_info
    }
