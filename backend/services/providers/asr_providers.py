import time
import numpy as np
import torch
from typing import Optional, Dict, Any
from transformers import pipeline, AutoModel
from backend.services.providers.base import BaseASRProvider
from backend.utils.logging_config import get_logger

logger = get_logger()

# Map common ISO 639-1 / Bhashini language codes to standard language names for Whisper
WHISPER_LANG_MAP = {
    "hi": "hindi", "hin": "hindi", "hin_Deva": "hindi", "hindi": "hindi",
    "ta": "tamil", "tam": "tamil", "tam_Taml": "tamil", "tamil": "tamil",
    "te": "telugu", "tel": "telugu", "tel_Telu": "telugu", "telugu": "telugu",
    "kn": "kannada", "kan": "kannada", "kan_Knda": "kannada", "kannada": "kannada",
    "bn": "bengali", "ben": "bengali", "ben_Beng": "bengali", "bengali": "bengali",
    "mr": "marathi", "mar": "marathi", "mar_Deva": "marathi", "marathi": "marathi",
    "gu": "gujarati", "guj": "gujarati", "guj_Gujr": "gujarati", "gujarati": "gujarati",
    "pa": "punjabi", "pan": "punjabi", "pan_Guru": "punjabi", "punjabi": "punjabi",
    "ml": "malayalam", "mal": "malayalam", "mal_Mlym": "malayalam", "malayalam": "malayalam",
    "en": "english", "eng": "english", "eng_Latn": "english", "english": "english"
}


class WhisperASRProvider(BaseASRProvider):
    """Provider for Hugging Face Whisper ASR models (e.g. openai/whisper-*, vasista22/whisper-tamil-small)."""

    def __init__(self):
        self.model_id = None
        self.device = None
        self.pipeline = None

    def load(self, model_id: str, device: torch.device, hf_token: Optional[str] = None):
        self.model_id = model_id
        self.device = device
        logger.info(f"Loading Whisper ASR model provider: '{model_id}' on {device}")
        self.pipeline = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=device,
            token=hf_token
        )
        logger.info(f"Whisper ASR model provider '{model_id}' loaded successfully.")

    def transcribe(self, waveform_tensor: torch.Tensor, language: str = "hi") -> Dict[str, Any]:
        if not self.pipeline:
            raise RuntimeError(f"Whisper ASR provider for '{self.model_id}' is not loaded.")

        audio_array = waveform_tensor.numpy()
        lang_normalized = language.strip().lower() if language else "hi"
        lang_full = WHISPER_LANG_MAP.get(lang_normalized, lang_normalized)

        try:
            forced_ids = self.pipeline.tokenizer.get_decoder_prompt_ids(language=lang_full, task="transcribe")
            gen_kwargs = {"forced_decoder_ids": forced_ids}
        except Exception as e:
            logger.warning(f"Could not get forced_decoder_ids for language '{lang_full}': {e}")
            gen_kwargs = {"task": "transcribe"}

        logger.info(f"Running Whisper transcription with generate_kwargs: {gen_kwargs}")

        result = self.pipeline(audio_array, generate_kwargs=gen_kwargs)
        text = result.get("text", "").strip()

        confidence = None
        if "chunks" in result:
            confidences = [c.get("confidence") for c in result["chunks"] if c.get("confidence") is not None]
            if confidences:
                confidence = float(np.mean(confidences))

        return {
            "text": text if text else "[Unintelligible speech]",
            "confidence": confidence,
            "engine": f"Whisper ({self.model_id})"
        }


class IndicConformerASRProvider(BaseASRProvider):
    """Provider for AI4Bharat IndicConformer ASR models."""

    def __init__(self):
        self.model_id = None
        self.device = None
        self.model = None

    def load(self, model_id: str, device: torch.device, hf_token: Optional[str] = None):
        self.model_id = model_id
        self.device = device
        logger.info(f"Loading IndicConformer ASR provider: '{model_id}' on {device}")
        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            token=hf_token
        ).to(device)
        logger.info(f"IndicConformer ASR provider '{model_id}' loaded successfully.")

    def transcribe(self, waveform_tensor: torch.Tensor, language: str = "hi") -> Dict[str, Any]:
        if not self.model:
            raise RuntimeError(f"IndicConformer provider for '{self.model_id}' is not loaded.")

        wav_2d = waveform_tensor.unsqueeze(0).to(self.device)
        lang_code = language.split("_")[0] if "_" in language else language

        with torch.no_grad():
            text = self.model(wav_2d, lang=lang_code, decoding="ctc")

        return {
            "text": text if text else "[Unintelligible speech]",
            "confidence": None,
            "engine": f"IndicConformer ({self.model_id})"
        }


class MockASRProvider(BaseASRProvider):
    """Mock ASR Provider for fast testing and offline mode."""

    def __init__(self):
        self.model_id = "mock-asr"

    def load(self, model_id: str, device: torch.device, hf_token: Optional[str] = None):
        self.model_id = model_id
        logger.info("Mock ASR provider ready.")

    def transcribe(self, waveform_tensor: torch.Tensor, language: str = "hi") -> Dict[str, Any]:
        time.sleep(0.1)
        mock_responses = {
            "ta": "வணக்கம் நீங்கள் எப்படி இருக்கிறீர்கள்",
            "te": "నమస్కారం మీరు ఎలా ఉన్నారు",
            "kn": "ನಮಸ್ಕಾರ ನೀವು ಹೇಗಿದ್ದೀರಿ",
            "hi": "नमस्ते आप कैसे हैं"
        }
        lang_code = language.split("_")[0] if language else "hi"
        text = mock_responses.get(lang_code, "नमस्ते आप कैसे हैं")
        return {"text": text, "confidence": 0.98, "engine": "Mock ASR"}
