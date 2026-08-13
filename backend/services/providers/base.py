from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import torch

class BaseASRProvider(ABC):
    """Abstract interface for Speech-to-Text (ASR) providers."""
    
    @abstractmethod
    def load(self, model_id: str, device: torch.device, hf_token: Optional[str] = None):
        """Loads model into memory."""
        pass

    @abstractmethod
    def transcribe(self, waveform_tensor: torch.Tensor, language: str = "hi") -> Dict[str, Any]:
        """
        Transcribes speech audio tensor to text.
        Returns dict with keys: 'text', 'confidence', 'engine'
        """
        pass


class BaseTranslationProvider(ABC):
    """Abstract interface for Text Translation providers."""

    @abstractmethod
    def load(self, model_id: str, device: torch.device, hf_token: Optional[str] = None):
        """Loads model into memory."""
        pass

    @abstractmethod
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translates text from source_lang to target_lang."""
        pass


class BaseTTSProvider(ABC):
    """Abstract interface for Text-to-Speech (TTS) providers."""

    @abstractmethod
    def load(self, model_id: str, device: torch.device, hf_token: Optional[str] = None):
        """Loads model into memory."""
        pass

    @abstractmethod
    def synthesize(self, text: str, ref_audio_bytes: Optional[bytes] = None, ref_text: Optional[str] = None) -> bytes:
        """Synthesizes text into audio WAV bytes."""
        pass
