from typing import Type, Dict
from backend.services.providers.base import BaseASRProvider, BaseTranslationProvider, BaseTTSProvider
from backend.services.providers.asr_providers import WhisperASRProvider, IndicConformerASRProvider, MockASRProvider
from backend.services.providers.translation_providers import IndicTrans2TranslationProvider, NLLBTranslationProvider, MockTranslationProvider
from backend.services.providers.tts_providers import IndicF5TTSProvider, MockTTSProvider
from backend.utils.logging_config import get_logger

logger = get_logger()


class ASRFactory:
    """Factory registry for Speech-to-Text (ASR) providers."""
    
    _registry: Dict[str, Type[BaseASRProvider]] = {
        "indic-conformer": IndicConformerASRProvider,
        "whisper": WhisperASRProvider,
    }

    @classmethod
    def register_provider(cls, keyword: str, provider_cls: Type[BaseASRProvider]):
        """Registers a new ASR model keyword to a Provider class."""
        cls._registry[keyword.lower()] = provider_cls
        logger.info(f"Registered new ASR provider for keyword '{keyword}': {provider_cls.__name__}")

    @classmethod
    def get_provider(cls, model_id: str, is_mock: bool = False) -> BaseASRProvider:
        if is_mock:
            return MockASRProvider()

        model_id_lower = model_id.lower()
        for keyword, provider_cls in cls._registry.items():
            if keyword in model_id_lower:
                logger.info(f"Matched ASR model '{model_id}' to provider class '{provider_cls.__name__}'")
                return provider_cls()

        # Default fallback for HF speech recognition models (e.g. Whisper variants)
        logger.info(f"No specific keyword match for ASR model '{model_id}', defaulting to WhisperASRProvider.")
        return WhisperASRProvider()


class TranslationFactory:
    """Factory registry for Text Translation providers."""

    _registry: Dict[str, Type[BaseTranslationProvider]] = {
        "indictrans2": IndicTrans2TranslationProvider,
        "nllb": NLLBTranslationProvider,
    }

    @classmethod
    def register_provider(cls, keyword: str, provider_cls: Type[BaseTranslationProvider]):
        """Registers a new Translation model keyword to a Provider class."""
        cls._registry[keyword.lower()] = provider_cls
        logger.info(f"Registered new Translation provider for keyword '{keyword}': {provider_cls.__name__}")

    @classmethod
    def get_provider(cls, model_id: str, is_mock: bool = False) -> BaseTranslationProvider:
        if is_mock:
            return MockTranslationProvider()

        model_id_lower = model_id.lower()
        for keyword, provider_cls in cls._registry.items():
            if keyword in model_id_lower:
                logger.info(f"Matched Translation model '{model_id}' to provider class '{provider_cls.__name__}'")
                return provider_cls()

        logger.info(f"No specific keyword match for Translation model '{model_id}', defaulting to NLLBTranslationProvider.")
        return NLLBTranslationProvider()


class TTSFactory:
    """Factory registry for Text-to-Speech (TTS) providers."""

    _registry: Dict[str, Type[BaseTTSProvider]] = {
        "indicf5": IndicF5TTSProvider,
        "f5": IndicF5TTSProvider,
    }

    @classmethod
    def register_provider(cls, keyword: str, provider_cls: Type[BaseTTSProvider]):
        """Registers a new TTS model keyword to a Provider class."""
        cls._registry[keyword.lower()] = provider_cls
        logger.info(f"Registered new TTS provider for keyword '{keyword}': {provider_cls.__name__}")

    @classmethod
    def get_provider(cls, model_id: str, is_mock: bool = False) -> BaseTTSProvider:
        if is_mock:
            return MockTTSProvider()

        model_id_lower = model_id.lower()
        for keyword, provider_cls in cls._registry.items():
            if keyword in model_id_lower:
                logger.info(f"Matched TTS model '{model_id}' to provider class '{provider_cls.__name__}'")
                return provider_cls()

        logger.info(f"No specific keyword match for TTS model '{model_id}', defaulting to IndicF5TTSProvider.")
        return IndicF5TTSProvider()
