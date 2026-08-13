import os
import re
import io
import time
import tempfile
import threading
import torch
import numpy as np
from typing import Optional
from backend.services.providers.base import BaseTTSProvider
from backend.utils.logging_config import get_logger
from backend.config import settings

logger = get_logger()


class IndicF5TTSProvider(BaseTTSProvider):
    """Provider for AI4Bharat IndicF5 TTS models."""

    def __init__(self):
        self.model_id = None
        self.device = None
        self.model = None
        self.lock = threading.Lock()

    def load(self, model_id: str, device: torch.device, hf_token: Optional[str] = None):
        # Add attribute check for PyTorch compatibility in IndicF5
        if not hasattr(torch, "xpu"):
            torch.xpu = None
        # Force CPU if MPS backend due to ComplexFloat support limitations
        self.device = torch.device("cpu") if device.type == "mps" else device
        logger.info(f"Loading IndicF5 TTS provider: '{model_id}' on {self.device}")

        from transformers import AutoModel
        from safetensors.torch import load_file
        from huggingface_hub import hf_hub_download

        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            token=hf_token
        ).to(self.device)

        safetensors_path = hf_hub_download(model_id, filename="model.safetensors", token=hf_token)
        state_dict = load_file(safetensors_path, device="cpu")
        cleaned_state_dict = {k.replace("._orig_mod.", "."): v for k, v in state_dict.items()}
        
        self.model.load_state_dict(cleaned_state_dict, strict=False)
        self.model.config.remove_sil = False

        try:
            import f5_tts.infer.utils_infer
            f5_tts.infer.utils_infer.nfe_step = 16
            logger.info("Monkey-patched f5_tts nfe_step to 16 for faster CPU inference")
        except Exception as err:
            logger.warning(f"Failed monkey-patching nfe_step: {err}")

        logger.info(f"IndicF5 TTS provider '{model_id}' loaded successfully.")

    def synthesize(self, text: str, ref_audio_bytes: Optional[bytes] = None, ref_text: Optional[str] = None) -> bytes:
        if not self.model:
            raise RuntimeError(f"IndicF5 TTS provider for '{self.model_id}' is not loaded.")

        with self.lock:
            temp_ref_path = None
            try:
                if ref_audio_bytes:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_ref:
                        temp_ref.write(ref_audio_bytes)
                        temp_ref_path = temp_ref.name
                    transcript = ref_text or ""
                else:
                    temp_ref_path = settings.DEFAULT_REF_AUDIO_PATH
                    transcript = settings.DEFAULT_REF_TEXT

                cleaned_text = re.sub(r'[।॥.,?!;:\"\'\-\(\)\[\]\{\}]', ' ', text)
                cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

                cleaned_transcript = re.sub(r'[।॥.,?!;:\"\'\-\(\)\[\]\{\}]', ' ', transcript)
                cleaned_transcript = re.sub(r'\s+', ' ', cleaned_transcript).strip()

                with torch.no_grad():
                    audio_array = self.model(
                        cleaned_text,
                        ref_audio_path=temp_ref_path,
                        ref_text=cleaned_transcript
                    )

                if audio_array.dtype == np.int16:
                    audio_array = audio_array.astype(np.float32) / 32768.0

                out_buf = io.BytesIO()
                sf.write(out_buf, np.array(audio_array, dtype=np.float32), 24000, format="WAV")
                return out_buf.getvalue()

            finally:
                if ref_audio_bytes and temp_ref_path and os.path.exists(temp_ref_path):
                    try:
                        os.remove(temp_ref_path)
                    except Exception as err:
                        logger.warning(f"Failed removing temp reference file: {err}")


class MockTTSProvider(BaseTTSProvider):
    """Mock TTS Provider for fast testing and offline mode."""

    def __init__(self):
        self.model_id = "mock-tts"

    def load(self, model_id: str, device: torch.device, hf_token: Optional[str] = None):
        self.model_id = model_id
        logger.info("Mock TTS provider ready.")

    def synthesize(self, text: str, ref_audio_bytes: Optional[bytes] = None, ref_text: Optional[str] = None) -> bytes:
        import soundfile as sf
        silent_audio = np.zeros(24000, dtype=np.float32)
        out_buf = io.BytesIO()
        sf.write(out_buf, silent_audio, 24000, format="WAV")
        return out_buf.getvalue()
