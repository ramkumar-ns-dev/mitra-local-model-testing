# Speech-to-Text (STT) & Translation Service

A robust, production-quality Python backend using **FastAPI** to perform Speech-to-Text (STT) and text translation. This service is optimized to load AI models on startup and run inference locally entirely offline.

## Project Structure
```
backend/
├── config.py             # Settings, limits, and configurations
├── main.py               # FastAPI initialization and middleware
├── requirements.txt      # Python library dependencies
├── run_backend.sh        # Startup script
├── api/
│   ├── __init__.py
│   └── routes.py         # /transcribe, /translate, /health routes
├── services/
│   ├── __init__.py
│   ├── audio_processor.py # Audio validation, conversion, and resampling
│   └── model_loader.py   # Model loading, warm-up, and fallback handling
└── utils/
    ├── __init__.py
    └── logging_config.py # Custom logger with Request ID tracing
```

---

## Local Setup Instructions

### Prerequisites
* **Python**: Version `3.9` to `3.12` is recommended.
* **FFmpeg**: Required for decoding non-WAV audio formats (like MP3, M4A, FLAC).
  * **macOS**: `brew install ffmpeg`
  * **Linux**: `sudo apt install ffmpeg`
  * **Windows**: Download binaries from [ffmpeg.org](https://ffmpeg.org/download.html) and add them to your system `PATH`.

### Installation Steps

#### 1. Setup Virtual Environment
Navigate to the `backend` folder:
```bash
cd backend
```

Create a virtual environment:
* **macOS / Linux**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
* **Windows (Command Prompt)**:
  ```cmd
  python -m venv venv
  venv\Scripts\activate.bat
  ```
* **Windows (PowerShell)**:
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```

#### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Configuration & Environment Variables

You can customize the service behavior using environment variables or a `.env` file in the `backend/` directory:

* `MOCK_MODELS`: If set to `True`, the server will skip downloading/loading PyTorch neural networks and use high-fidelity mock outputs. This allows instant setup and testing without downloading multiple gigabytes of weights. Defaults to `False`.
* `ASR_MODEL_ID`: The primary Hugging Face ASR model to load. Defaults to `ai4bharat/conformer-hi-gpu--t4`. Since that identifier represents a cloud-only Bhashini service ID, the system will automatically catch loading errors and fall back to `FALLBACK_ASR_MODEL_ID`.
* `FALLBACK_ASR_MODEL_ID`: Hugging Face model loaded if the primary fails. Defaults to `openai/whisper-tiny` (approx 70MB, runs extremely fast locally on CPU/macOS).
* `TRANSLATION_MODEL_ID`: The model used for translation. Defaults to `facebook/nllb-200-distilled-600M` (runs locally, fully supports NLLB-200 language tags).

Example `.env` file:
```env
MOCK_MODELS=False
FALLBACK_ASR_MODEL_ID=openai/whisper-tiny
TRANSLATION_MODEL_ID=facebook/nllb-200-distilled-600M
```

---

## Starting the Server

To start the server, run the following command from the `backend/` directory:
```bash
# Run with local development reload (reload disabled in production-quality model execution)
uvicorn main:app --host 127.0.0.1 --port 8000
```
Or use the provided startup script (macOS/Linux):
```bash
chmod +x run_backend.sh
./run_backend.sh
```

---

## Testing & Sample Curl Commands

### 1. Health Check
Checks if models are loaded and healthy:
```bash
curl -X GET http://localhost:8000/health
```
**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "details": {
    "asr_status": "ready",
    "asr_model": "openai/whisper-tiny",
    "translation_status": "ready",
    "translation_model": "facebook/nllb-200-distilled-600M",
    "device": "mps"
  }
}
```

### 2. Speech-to-Text Transcription (`/transcribe`)

#### Option A: Multipart Form-Data
Upload an audio file (supports `.wav`, `.mp3`, `.m4a`, `.flac`):
```bash
curl -X POST http://localhost:8000/transcribe \
  -F "audio=@sample.wav"
```

#### Option B: Audio URL (JSON)
Provide a URL to download and transcribe:
```bash
curl -X POST http://localhost:8000/transcribe \
  -H "Content-Type: application/json" \
  -d '{
    "audio_url": "https://raw.githubusercontent.com/benvandy/wav-samples/master/sin_1000Hz_-3dBFS_1s.wav"
  }'
```

#### Option C: Base64 Encoded Audio (JSON)
Provide base64-encoded audio data:
```bash
curl -X POST http://localhost:8000/transcribe \
  -H "Content-Type: application/json" \
  -d '{
    "audio_base64": "data:audio/wav;base64,UklGRiQAAABXQVZ..."
  }'
```

**Response**:
```json
{
  "success": true,
  "text": "नमस्ते आप कैसे हैं",
  "confidence": 0.9834
}
```

### 3. Translation (`/translate`)
Translates text using NLLB-200 language tags (compatible with your frontend service):
```bash
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, how are you?",
    "source_lang": "eng_Latn",
    "target_lang": "hin_Deva"
  }'
```
**Response**:
```json
{
  "translation": "नमस्ते, आप कैसे हैं?"
}
```

---

## Security & Reliability Features
1. **Size Limits**: Enforces a `50MB` file upload limit to prevent memory exhaustion.
2. **Duration Limits**: Enforces a `300s` (5-minute) audio duration limit to prevent inference time-outs.
3. **Format Validations**: Prevents processing of non-audio files by checking extensions and headers.
4. **Resampling**: Downmixes multi-channel audio to mono and resamples it to `16000Hz` in memory before passing to neural networks.
5. **Request Tracking**: Generates or forwards `X-Request-ID` headers to trace log metrics (durations, inference latencies, errors) on every request.
