import base64
import requests
import time
import psutil
from typing import Optional
from fastapi import APIRouter, Request, HTTPException, status
from pydantic import BaseModel, Field
from backend.services.audio_processor import AudioProcessor
from backend.services.model_loader import model_loader
from backend.utils.logging_config import get_logger

logger = get_logger()
router = APIRouter()

# Language Mapping (ISO 639-1 / Bhashini code -> NLLB-200 code)
LANG_MAP = {
    "as": "asm_Beng", "bn": "ben_Beng", "brx": "bod_Tibt", "doi": "doi_Deva",
    "en": "eng_Latn", "gu": "guj_Gujr", "hi": "hin_Deva", "kn": "kan_Knda",
    "ks": "kas_Arab", "kok": "kok_Deva", "mai": "mai_Deva", "ml": "mal_Mlym",
    "mni": "mni_Beng", "mr": "mar_Deva", "ne": "nep_Deva", "or": "ory_Orya",
    "pa": "pan_Guru", "sa": "san_Deva", "sat": "sat_Olch", "sd": "snd_Deva",
    "ta": "tam_Taml", "te": "tel_Telu", "ur": "urd_Aran"
}

# Translation Request Schema (Supports both NLLB tags and Angular 2-letter codes)
class TranslateRequest(BaseModel):
    text: str = Field(..., description="The text to translate")
    source_lang: Optional[str] = Field(None, description="Source language tag, e.g., eng_Latn")
    target_lang: Optional[str] = Field(None, description="Target language tag, e.g., hin_Deva")
    src_lang: Optional[str] = Field(None, description="Legacy source language tag, e.g., en")
    tgt_lang: Optional[str] = Field(None, description="Legacy target language tag, e.g., hi")

@router.post("/transcribe")
async def transcribe(request: Request):
    """
    Speech-to-Text Endpoint.
    Supports both standard field names ('audio') and Angular frontend field names ('file').
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
            # Support both 'audio' (standard curl) and 'file' (Angular frontend)
            upload_file = form.get("audio") or form.get("file")
            
            if not upload_file:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, 
                    detail="Missing audio file field ('audio' or 'file') in form data"
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
                    response = requests.get(audio_url, timeout=15.0)
                    response.raise_for_status()
                    file_bytes = response.content
                    file_size = len(file_bytes)
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
                    if "," in audio_base64:
                        audio_base64 = audio_base64.split(",")[1]
                    file_bytes = base64.b64decode(audio_base64)
                    file_size = len(file_bytes)
                    filename = body.get("filename", "audio.wav")
                except Exception as e:
                    logger.error(f"Failed to decode base64 audio: {e}")
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Failed to decode base64 audio."
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
            "text": inference_result["text"],
            "metrics": {
                "latency_ms": total_time_ms,
                "engine": inference_result["engine"]
            }
        }
        
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
    Supports both NLLB-200 tags (source_lang) and legacy ISO-639 codes (src_lang).
    """
    start_time = time.time()
    
    # Extract languages dynamically
    src = body.source_lang or body.src_lang
    tgt = body.target_lang or body.tgt_lang
    
    if not src or not tgt:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing source/target language parameters."
        )
        
    # Map legacy 2-letter codes to NLLB codes if present in map
    src_mapped = LANG_MAP.get(src, src)
    tgt_mapped = LANG_MAP.get(tgt, tgt)
    
    logger.info(f"Translation request: src={src} ({src_mapped}), tgt={tgt} ({tgt_mapped}), text_len={len(body.text)}")
    
    try:
        translated_text = model_loader.translate(
            text=body.text,
            source_lang=src_mapped,
            target_lang=tgt_mapped
        )
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        logger.info(f"Translation completed in {processing_time_ms}ms")
        
        return {
            "translation": translated_text,
            "metrics": {
                "latency_ms": processing_time_ms,
                "engine": f"Local IndicTrans2 ({model_loader.translation_model_name})" if model_loader.translation_status == "ready" else "Simulated Engine"
            }
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
    """
    status_info = model_loader.get_status()
    asr_ready = status_info["asr_status"] in ("ready", "ready_mock")
    translation_ready = status_info["translation_status"] in ("ready", "ready_mock")
    is_healthy = asr_ready and translation_ready
    
    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "model_loaded": is_healthy,
        "details": status_info
    }

# Legacy Endpoints for Angular Frontend Compatibility

@router.get("/status")
def get_legacy_status():
    """
    Legacy GET /status endpoint polled by the Angular frontend.
    """
    status_info = model_loader.get_status()
    
    # Gather system stats
    cpu_percent = psutil.cpu_percent()
    memory_info = psutil.virtual_memory()
    
    return {
        "status": "online",
        "device": status_info["device"],
        "torch_available": True,
        "models": {
            "indic_conformer_asr": status_info["asr_status"],
            "indictrans2_en_indic": status_info["translation_status"],
            "indictrans2_indic_en": status_info["translation_status"]
        },
        "loading": (status_info["asr_status"] == "loading") or (status_info["translation_status"] == "loading"),
        "error": "",
        "system": {
            "cpu_usage_percent": cpu_percent,
            "memory_usage_percent": memory_info.percent,
            "memory_available_gb": round(memory_info.available / (1024**3), 2),
            "memory_total_gb": round(memory_info.total / (1024**3), 2)
        }
    }

class LoadRequest(BaseModel):
    hf_token: Optional[str] = None

@router.post("/load")
def trigger_legacy_load(request: LoadRequest):
    """
    Legacy POST /load endpoint called by the Angular frontend.
    """
    return {"status": "already_loading" if model_loader.asr_status == "loading" else "ready"}
