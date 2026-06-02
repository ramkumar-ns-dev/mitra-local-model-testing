# Mitra Translate: AI4Bharat Translation & Voice Platform

Mitra Translate is a translation application built with a modern **Angular frontend** and a **Python FastAPI backend**. It leverages AI4Bharat's state-of-the-art open-source speech and translation models to enable bilingual text and voice translation between English and 22 official Indian languages.

---

## Key Features

1. **AI4Bharat Speech & Translation Models**:
   - **ASR (Speech-to-Text)**: Powered by the AI4Bharat `indic-conformer-600m-multilingual` model.
   - **Translation**: Powered by the distilled AI4Bharat `indictrans2-en-indic-dist-200M` and `indictrans2-indic-en-dist-200M` models (optimized for CPU/Mac execution).
2. **Three Integration Modes**:
   - **Local Model Mode (Offline)**: Performs voice transcription and translation fully locally on your system using loaded AI4Bharat weights.
   - **Bhashini Cloud Mode**: Connects directly to the official government Bhashini ULCA API (requires your own credentials, saved locally).
   - **Simulated Demo Mode**: Allow instant UI evaluation (pre-configured translation lookups and simulated ASR waveforms) without any setup.
3. **Working Efficiency Metrics**:
   - Live telemetry dashboard capturing translation round-trip latency, word count, character throughput (characters/sec), and ASR decoding speed.
   - Real-time CPU and Memory utilization of the local model server.
4. **Interactive Dashboard Utilities**:
   - Record voice inputs directly via the browser microphone.
   - Text-to-Speech (TTS) readback for translated text (using Web Speech Synthesis).
   - persistent translation history logger with export options (JSON / CSV).
   - Glassmorphic dark mode styling.

---

## Project Structure

```
├── backend/
│   ├── main.py            # FastAPI backend with AI4Bharat endpoints
│   ├── requirements.txt   # Python ML & web dependencies
│   └── run_backend.sh     # Setup script to boot python server in venv
├── frontend/
│   ├── src/               # Angular client source code
│   ├── angular.json       # App configuration and budgets
│   └── package.json       # Angular packages and run scripts
└── README.md              # Documentation
```

---

## Getting Started

### Prerequisites

1. **Python 3.12+** (with pip)
2. **Node.js v20+** (with npm)
3. **Chrome / Safari** (supporting Web Speech API and MediaRecorder)

---

### Step 1: Start the Python Backend

The backend server loads the models locally. Run the automated script which sets up a python virtual environment, installs dependencies, and runs the server:

```bash
# Navigate to the backend directory and run the launcher
cd backend
./run_backend.sh
```

- On first run, starting the FastAPI server will take time to download the model weights (about 2.5 GB total) from Hugging Face.
- Once running, the backend API will be available at `http://127.0.0.1:8000`.

---

### Step 2: Start the Angular Frontend

In a separate terminal window, build and start the Angular development server:

```bash
# Navigate to the frontend directory
cd frontend

# Start the Angular development server
npm run start
```

- Open [http://localhost:4200](http://localhost:4200) in your web browser.

---

## Using the Application

1. **Demo Mode (Default)**: Use the app immediately. Type "Hello", "How are you", or "Thank you" and hit **Translate**, or click the Microphone button to record and see how the UI logs the translation speed, throughput, and history.
2. **Local AI4Bharat Mode**: 
   - Boot the Python backend.
   - Select **Local Model** under *Setup*.
   - If not loaded, click **Load Local Models**. The UI will track the loading process.
   - Speak or type text in Hindi, Tamil, Telugu, etc., and translate it offline.
3. **Bhashini Cloud Mode**: 
   - Select **Bhashini Cloud**.
   - Input your `User ID`, `API Key`, and `Auth Token` generated from the Bhashini developer portal.
   - Send requests directly to the government translation API.
4. **History Logs**: Click the **Export CSV** or **Export JSON** buttons at the bottom to download a report of your model's working efficiency.
