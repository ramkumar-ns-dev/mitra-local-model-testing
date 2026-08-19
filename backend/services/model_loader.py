import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
import torch
import numpy as np

# Monkey-patch torch.compile to prevent Dynamo Python 3.12+ compatibility errors in IndicF5
if sys.version_info >= (3, 12):
    torch.compile = lambda model, *args, **kwargs: model

from typing import Optional, Dict, Any
from backend.config import settings
from backend.utils.logging_config import get_logger
from backend.services.providers.factory import ASRFactory, TranslationFactory, TTSFactory
from backend.services.providers.base import BaseASRProvider, BaseTranslationProvider, BaseTTSProvider

logger = get_logger()


class ModelLoader:
    """
    ModelLoader Service Singleton.
    Orchestrates ASR, Translation, and TTS providers using Factory pattern.
    Supports multi-model ASR routing per language.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ModelLoader, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.device = self._get_device()
        self.asr_status = "not_loaded"
        self.translation_status = "not_loaded"
        self.tts_status = "not_loaded"

        self.asr_model_name = settings.ASR_MODEL_ID
        self.translation_model_name = settings.TRANSLATION_MODEL_ID
        self.tts_model_name = settings.TTS_MODEL_ID

        self.asr_provider: Optional[BaseASRProvider] = None
        self.asr_providers: Dict[str, BaseASRProvider] = {}
        self.translation_provider: Optional[BaseTranslationProvider] = None
        self.tts_provider: Optional[BaseTTSProvider] = None

        self._initialized = True

    def _get_device(self) -> torch.device:
        """Determines the device to run inference on."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Using NVIDIA CUDA for inference")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using Apple Silicon MPS for inference")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU for inference")
        return device

    def _get_or_load_asr_provider(self, model_id: str) -> BaseASRProvider:
        """Retrieves or dynamically loads an ASR provider for a given model ID."""
        if model_id in self.asr_providers:
            return self.asr_providers[model_id]

        if settings.MOCK_MODELS:
            provider = ASRFactory.get_provider(model_id, is_mock=True)
            self.asr_providers[model_id] = provider
            return provider

        try:
            logger.info(f"Loading ASR model on demand: {model_id}")
            if "gpu--t4" in model_id:
                raise ValueError(f"{model_id} is a Bhashini API service ID and cannot be loaded as a local HF repository.")

            provider = ASRFactory.get_provider(model_id)
            provider.load(model_id, self.device, hf_token=settings.HF_TOKEN)
            self.asr_providers[model_id] = provider
            logger.info(f"ASR provider '{model_id}' loaded and cached successfully.")
            return provider
        except Exception as e:
            logger.warning(f"Failed to load ASR model '{model_id}': {e}. Attempting fallback to '{settings.FALLBACK_ASR_MODEL_ID}'...")
            try:
                fallback_id = settings.FALLBACK_ASR_MODEL_ID
                if fallback_id in self.asr_providers:
                    return self.asr_providers[fallback_id]
                fb_provider = ASRFactory.get_provider(fallback_id)
                fb_provider.load(fallback_id, self.device, hf_token=settings.HF_TOKEN)
                self.asr_providers[fallback_id] = fb_provider
                return fb_provider
            except Exception as fe:
                logger.error(f"Failed to load ASR fallback model: {fe}. Falling back to MOCK provider.")
                mock_provider = ASRFactory.get_provider("mock", is_mock=True)
                self.asr_providers[model_id] = mock_provider
                return mock_provider

    def load_models(self):
        """Loads ASR, Translation, and TTS models via Provider Factories."""
        # Always refresh configured model IDs from settings
        self.asr_model_name = settings.ASR_MODEL_ID
        self.translation_model_name = settings.TRANSLATION_MODEL_ID
        self.tts_model_name = settings.TTS_MODEL_ID
        self.asr_providers.clear()

        if settings.MOCK_MODELS:
            logger.warning("MOCK_MODELS is enabled. Running in simulated mode without loading neural networks.")
            self.asr_provider = ASRFactory.get_provider(self.asr_model_name, is_mock=True)
            self.asr_providers[self.asr_model_name] = self.asr_provider
            self.translation_provider = TranslationFactory.get_provider(self.translation_model_name, is_mock=True)
            self.tts_provider = TTSFactory.get_provider(self.tts_model_name, is_mock=True)
            self.asr_status = "ready_mock"
            self.translation_status = "ready_mock"
            self.tts_status = "ready_mock"
            return

        # 1. Load Speech-to-Text (ASR) Primary / Default Provider
        try:
            self.asr_status = "loading"
            primary_model = settings.get_asr_model_for_language("default")
            logger.info(f"Loading primary/default ASR model: {primary_model}")
            self.asr_provider = self._get_or_load_asr_provider(primary_model)
            self.asr_status = "ready"
            logger.info(f"Primary ASR provider '{primary_model}' loaded successfully. Additional per-language models will load on-demand.")

        except Exception as e:
            logger.warning(f"Error during primary ASR model loading: {e}. Attempting fallback.")
            self.asr_provider = self._get_or_load_asr_provider(settings.FALLBACK_ASR_MODEL_ID)
            self.asr_status = "ready"

        # 2. Load Translation Provider
        try:
            self.translation_status = "loading"
            logger.info(f"Loading translation model: {self.translation_model_name}")
            self.translation_provider = TranslationFactory.get_provider(self.translation_model_name)
            self.translation_provider.load(self.translation_model_name, self.device, hf_token=settings.HF_TOKEN)
            self.translation_status = "ready"
            logger.info(f"Translation provider '{self.translation_model_name}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load translation model '{self.translation_model_name}': {e}. Falling back to MOCK mode for translation.")
            self.translation_provider = TranslationFactory.get_provider("mock", is_mock=True)
            self.translation_status = "ready_mock"

        # 3. Load Text-to-Speech (TTS) Provider
        try:
            self.tts_status = "loading"
            logger.info(f"Loading TTS model: {self.tts_model_name}")
            self.tts_provider = TTSFactory.get_provider(self.tts_model_name)
            self.tts_provider.load(self.tts_model_name, self.device, hf_token=settings.HF_TOKEN)
            self.tts_status = "ready"
            logger.info(f"TTS provider '{self.tts_model_name}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load TTS model '{self.tts_model_name}': {e}. Falling back to MOCK mode for TTS.")
            self.tts_provider = TTSFactory.get_provider("mock", is_mock=True)
            self.tts_status = "ready_mock"

    def transcribe(self, waveform_tensor: torch.Tensor, language: str = "hi") -> Dict[str, Any]:
        """Runs speech-to-text inference using language-specific or active ASR provider."""
        model_id = settings.get_asr_model_for_language(language)
        provider = self._get_or_load_asr_provider(model_id)
        if not provider:
            if self.asr_provider:
                provider = self.asr_provider
            else:
                raise RuntimeError("No ASR model is initialized or loaded.")
        return provider.transcribe(waveform_tensor, language=language)

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translates text from source_lang to target_lang using active Translation provider."""
        if not self.translation_provider:
            raise RuntimeError("Translation model is not initialized or loaded.")
        return self.translation_provider.translate(text, source_lang=source_lang, target_lang=target_lang)

    def synthesize(self, text: str, ref_audio_bytes: Optional[bytes] = None, ref_text: Optional[str] = None) -> bytes:
        """Synthesizes speech using active TTS provider."""
        if not self.tts_provider:
            raise RuntimeError("TTS model is not initialized or loaded.")
        return self.tts_provider.synthesize(text, ref_audio_bytes=ref_audio_bytes, ref_text=ref_text)

    def get_status(self) -> dict:
        """Returns model loading status and devices."""
        return {
            "asr_status": self.asr_status,
            "asr_model": self.asr_model_name,
            "asr_models_by_language": settings.ASR_MODELS,
            "loaded_asr_models": list(self.asr_providers.keys()),
            "translation_status": self.translation_status,
            "translation_model": self.translation_model_name,
            "tts_status": self.tts_status,
            "tts_model": self.tts_model_name,
            "device": str(self.device)
        }


# Global singleton
model_loader = ModelLoader()
