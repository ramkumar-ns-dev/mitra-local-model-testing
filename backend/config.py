import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)


class Settings:
    # API Settings
    API_TITLE: str = "AI4Bharat STT & Translation Service"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")

    # Hardware & Performance Settings
    DEVICE: str = os.getenv("DEVICE", "auto")  # 'auto', 'cpu', 'cuda', or 'mps'
    TORCH_THREADS: int = int(os.getenv("TORCH_THREADS", 4))
    MOCK_MODELS: bool = os.getenv("MOCK_MODELS", "False").lower() in ("true", "1", "yes")

    # ASR Model Settings
    ASR_MODEL_ID: str = os.getenv("ASR_MODEL_ID", "vasista22/whisper-tamil-small")
    FALLBACK_ASR_MODEL_ID: str = os.getenv("FALLBACK_ASR_MODEL_ID", "openai/whisper-tiny")
    
    # Per-Language ASR Model Map from environment JSON string
    ASR_MODELS_JSON: str = os.getenv("ASR_MODELS_JSON", "")
    
    @property
    def ASR_MODELS(self) -> dict:
        """Parses ASR_MODELS_JSON or falls back to single ASR_MODEL_ID."""
        if self.ASR_MODELS_JSON:
            try:
                import json
                parsed = json.loads(self.ASR_MODELS_JSON)
                if isinstance(parsed, dict):
                    return {k.lower(): v for k, v in parsed.items()}
            except Exception:
                pass
        return {"default": self.ASR_MODEL_ID}

    def get_asr_model_for_language(self, lang: str = "") -> str:
        """Resolves the configured ASR model ID for a specific language."""
        models = self.ASR_MODELS
        if not lang:
            return models.get("default", self.ASR_MODEL_ID)

        lang_lower = lang.strip().lower()
        # Direct lookup (e.g. 'ta', 'hi', 'tam_taml')
        if lang_lower in models:
            return models[lang_lower]

        # Extract 2-letter or root prefix (e.g. 'tam_Taml' -> 'ta', 'eng_Latn' -> 'en')
        code_prefix = lang_lower.split("_")[0].split("-")[0]
        if code_prefix in models:
            return models[code_prefix]

        # Common 3-letter to 2-letter fallback lookup
        lang_3_to_2 = {
            "tam": "ta", "hin": "hi", "eng": "en", "tel": "te",
            "kan": "kn", "mar": "mr", "ben": "bn", "guj": "gu",
            "pan": "pa", "mal": "ml", "urd": "ur", "ory": "or",
            "asm": "as", "san": "sa", "nep": "ne"
        }
        short_code = lang_3_to_2.get(code_prefix)
        if short_code and short_code in models:
            return models[short_code]

        return models.get("default", self.ASR_MODEL_ID)

    # Translation Model Settings
    TRANSLATION_MODEL_ID: str = os.getenv("TRANSLATION_MODEL_ID", "ai4bharat/indictrans2-en-indic-dist-200M")
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")

    # TTS Model Settings
    TTS_MODEL_ID: str = os.getenv("TTS_MODEL_ID", "ai4bharat/IndicF5")
    DEFAULT_REF_AUDIO_PATH: str = os.getenv(
        "DEFAULT_REF_AUDIO_PATH",
        str(Path(__file__).parent / "resources" / "default_voice.wav")
    )
    DEFAULT_REF_TEXT: str = os.getenv(
        "DEFAULT_REF_TEXT",
        "शिक्षा हमारे जीवन का एक महत्वपूर्ण आधार है यहां न केवल हमें ज्ञान प्रद"
    )

    # Security & Processing Limits
    MAX_FILE_SIZE_BYTES: int = 50 * 1024 * 1024  # 50 MB
    MAX_AUDIO_DURATION_SECONDS: float = 300.0   # 5 minutes
    
    ALLOWED_EXTENSIONS: set = {".wav", ".mp3", ".m4a", ".flac"}
    ALLOWED_MIME_TYPES: set = {
        "audio/wav", "audio/x-wav", 
        "audio/mpeg", "audio/mp3", 
        "audio/m4a", "audio/x-m4a",
        "audio/flac", "audio/x-flac"
    }

    # Audio Resampling
    TARGET_SAMPLE_RATE: int = 16000


# Initialize settings
settings = Settings()
