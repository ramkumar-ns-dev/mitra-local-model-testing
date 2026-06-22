import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from local .env file
load_dotenv()

class Settings:
    # API Settings
    API_TITLE: str = "AI4Bharat STT & Translation Service"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    # Mock Mode Settings
    # If True, runs mock inference for fast testing without loading heavy PyTorch models.
    # If False, loads the actual models.
    MOCK_MODELS: bool = os.getenv("MOCK_MODELS", "False").lower() in ("true", "1", "yes")

    # ASR Model Settings
    ASR_MODEL_ID: str = os.getenv("ASR_MODEL_ID", "ai4bharat/indic-conformer-600m-multilingual")
    FALLBACK_ASR_MODEL_ID: str = os.getenv("FALLBACK_ASR_MODEL_ID", "openai/whisper-tiny")
    
    # Translation Model Settings
    TRANSLATION_MODEL_ID: str = os.getenv("TRANSLATION_MODEL_ID", "ai4bharat/indictrans2-en-indic-dist-200M")
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")

    # TTS Model Settings (IndicF5)
    TTS_MODEL_ID: str = os.getenv("TTS_MODEL_ID", "ai4bharat/IndicF5")
    DEFAULT_REF_AUDIO_PATH: str = os.getenv("DEFAULT_REF_AUDIO_PATH", str(Path(__file__).parent / "resources" / "default_voice.wav"))
    DEFAULT_REF_TEXT: str = os.getenv("DEFAULT_REF_TEXT", "शिक्षा हमारे जीवन का एक महत्वपूर्ण आधार है यहां न केवल हमें ज्ञान प्रद")

    # Security & Limits
    # 50 MB maximum audio file size limit
    MAX_FILE_SIZE_BYTES: int = 50 * 1024 * 1024
    # 5 minutes maximum audio duration limit
    MAX_AUDIO_DURATION_SECONDS: float = 300.0
    
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
