import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { interval, Subscription } from 'rxjs';
import { catchError } from 'rxjs/operators';

interface Language {
  code: string;
  name: string;
  nativeName: string;
  ttsLangCode?: string; // Voice code for speech synthesis
}

interface HistoryEntry {
  timestamp: Date;
  srcLang: string;
  tgtLang: string;
  srcText: string;
  tgtText: string;
  latency: number;
  cps: number;
  engine: string;
  costInr?: number;
  inputTokens?: number;
  outputTokens?: number;
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent implements OnInit, OnDestroy {
  // App state - test change
  engineMode: 'local' | 'bhashini' | 'demo' | 'cloud_llm' = 'local';
  serverStatus: 'online' | 'offline' | 'loading' = 'offline';
  serverStatusText = 'Disconnected';
  isBackendLoading = false;
  areLocalModelsLoaded = false;
  hfToken = '';
  deviceTarget = 'CPU';
  configuredModels = {
    asr: 'vasista22/whisper-tamil-small',
    translation: 'ai4bharat/indictrans2-indic-en-1B',
    tts: 'ai4bharat/IndicF5'
  };

  // Translation workspace fields
  sourceText = '';
  targetText = '';
  sourceLang = 'hi';
  targetLang = 'en';
  isTranslating = false;
  showCopyToast = false;

  // Voice Recording state
  isRecording = false;
  mediaRecorder: MediaRecorder | null = null;
  audioChunks: Blob[] = [];
  wavRecorder: WavRecorder | null = null;

  // Telemetry metrics
  metrics = {
    latency: 0,
    cps: 0,
    asrLatency: 0,
    engine: 'Simulated Engine',
    costInr: 0,
    inputTokens: 0,
    outputTokens: 0
  };

  sysMetrics = {
    cpuUsage: '0%',
    ramUsage: '0 MB / 0 GB'
  };

  modelStates = {
    asr: 'not_loaded',
    en_indic: 'not_loaded',
    indic_en: 'not_loaded',
    tts: 'not_loaded'
  };

  // TTS configuration
  ttsMode: 'indicf5' | 'native' = 'indicf5';
  ttsVoiceMode: 'default' | 'clone' = 'default';
  refAudioFile: File | null = null;
  refTranscript: string = '';
  isSynthesizing = false;
  activeTtsSide: 'source' | 'target' | null = null;
  synthesisLatency = 0;
  private currentAudio: HTMLAudioElement | null = null;

  // Credentials for Bhashini Cloud API
  bhashiniConfig = {
    userId: '',
    apiKey: '',
    authToken: ''
  };

  // Bhashini Rates Configuration (INR per 1,000 characters)
  bhashiniRates = {
    translationRatePer1000: 1.0
  };

  // Cloud LLM Configuration
  selectedLlmModel = 'gemini-1.5-flash';
  usdToInrRate = 84.0;
  llmKeys = {
    openai: '',
    gemini: '',
    anthropic: '',
    deepseek: ''
  };

  // Translation history
  history: HistoryEntry[] = [];

  // Supported languages list (22 official Indian languages + English)
  languages: Language[] = [
    { code: 'en', name: 'English', nativeName: 'English', ttsLangCode: 'en-US' },
    { code: 'hi', name: 'Hindi', nativeName: 'हिन्दी', ttsLangCode: 'hi-IN' },
    { code: 'ta', name: 'Tamil', nativeName: 'தமிழ்', ttsLangCode: 'ta-IN' },
    { code: 'te', name: 'Telugu', nativeName: 'తెలుగు', ttsLangCode: 'te-IN' },
    { code: 'kn', name: 'Kannada', nativeName: 'ಕನ್ನಡ', ttsLangCode: 'kn-IN' },
    { code: 'ml', name: 'Malayalam', nativeName: 'മലയാളം', ttsLangCode: 'ml-IN' },
    { code: 'mr', name: 'Marathi', nativeName: 'மराठी', ttsLangCode: 'mr-IN' },
    { code: 'gu', name: 'Gujarati', nativeName: 'ગુજરાતી', ttsLangCode: 'gu-IN' },
    { code: 'bn', name: 'Bengali', nativeName: 'বাংলা', ttsLangCode: 'bn-IN' },
    { code: 'pa', name: 'Punjabi', nativeName: 'ਪੰਜਾਬੀ', ttsLangCode: 'pa-IN' },
    { code: 'or', name: 'Odia', nativeName: 'ଓଡ଼ିଆ' },
    { code: 'as', name: 'Assamese', nativeName: 'অসমীয়া' },
    { code: 'sa', name: 'Sanskrit', nativeName: 'संस्कृतं' },
    { code: 'ur', name: 'Urdu', nativeName: 'اردو' },
    { code: 'ne', name: 'Nepali', nativeName: 'नेपाली' },
    { code: 'ks', name: 'Kashmiri', nativeName: 'کٲشُر' },
    { code: 'kok', name: 'Konkani', nativeName: 'कोंकणी' },
    { code: 'sd', name: 'Sindhi', nativeName: 'سنڌي' },
    { code: 'doi', name: 'Dogri', nativeName: 'डोगरी' },
    { code: 'mai', name: 'Maithili', nativeName: 'मैथिली' },
    { code: 'mni', name: 'Manipuri', nativeName: 'মণিপুরী' },
    { code: 'sat', name: 'Santali', nativeName: 'ᱥᱟᱱᱛᱟᱲᱤ' },
    { code: 'brx', name: 'Bodo', nativeName: 'बर\'' }
  ];

  // Observables polling backend status
  private statusSubscription: Subscription | null = null;
  backendBaseUrl = 'http://127.0.0.1:8000';

  get cleanBackendBaseUrl(): string {
    const raw = (this.backendBaseUrl || 'http://127.0.0.1:8000').trim();
    return raw.replace(/\/+$/, '');
  }

  constructor(private http: HttpClient) {}

  ngOnInit() {
    // Load config from localStorage
    const savedConfig = localStorage.getItem('mitra_bhashini_config');
    if (savedConfig) {
      this.bhashiniConfig = JSON.parse(savedConfig);
    }

    const savedLlmKeys = localStorage.getItem('mitra_llm_keys');
    if (savedLlmKeys) {
      this.llmKeys = JSON.parse(savedLlmKeys);
    }

    this.selectedLlmModel = localStorage.getItem('mitra_llm_model') || 'gemini-1.5-flash';

    const savedRate = localStorage.getItem('mitra_usd_to_inr_rate');
    if (savedRate) {
      this.usdToInrRate = parseFloat(savedRate);
    }

    const savedHistory = localStorage.getItem('mitra_translation_history');
    if (savedHistory) {
      this.history = JSON.parse(savedHistory);
    }

    const savedBhashiniRates = localStorage.getItem('mitra_bhashini_rates');
    if (savedBhashiniRates) {
      this.bhashiniRates = JSON.parse(savedBhashiniRates);
    }

    const savedMode = localStorage.getItem('mitra_engine_mode');
    if (savedMode) {
      this.engineMode = savedMode as any;
    }

    this.hfToken = localStorage.getItem('mitra_hf_token') || '';

    const savedUrl = localStorage.getItem('mitra_backend_url');
    if (savedUrl && !savedUrl.includes('elevate-apis.shikshalokam.org')) {
      this.backendBaseUrl = savedUrl;
    } else {
      this.backendBaseUrl = 'http://127.0.0.1:8000';
      localStorage.setItem('mitra_backend_url', 'http://127.0.0.1:8000');
    }

    // Start polling python backend status
    this.checkBackendStatus();
    this.statusSubscription = interval(60000).subscribe(() => {
      this.checkBackendStatus();
    });
  }

  saveLlmKeys() {
    localStorage.setItem('mitra_llm_keys', JSON.stringify(this.llmKeys));
  }

  saveLlmModel() {
    localStorage.setItem('mitra_llm_model', this.selectedLlmModel);
  }

  saveUsdToInrRate() {
    localStorage.setItem('mitra_usd_to_inr_rate', this.usdToInrRate.toString());
  }

  saveBhashiniRates() {
    localStorage.setItem('mitra_bhashini_rates', JSON.stringify(this.bhashiniRates));
  }

  ngOnDestroy() {
    if (this.statusSubscription) {
      this.statusSubscription.unsubscribe();
    }
  }

  setEngineMode(mode: 'local' | 'bhashini' | 'demo' | 'cloud_llm') {
    this.engineMode = mode;
    localStorage.setItem('mitra_engine_mode', mode);
    this.updateMetricsEngine();
  }

  saveBhashiniKeys() {
    localStorage.setItem('mitra_bhashini_config', JSON.stringify(this.bhashiniConfig));
  }

  saveHfToken() {
    localStorage.setItem('mitra_hf_token', this.hfToken);
  }

  saveBackendBaseUrl() {
    localStorage.setItem('mitra_backend_url', this.backendBaseUrl);
    this.checkBackendStatus();
  }

  updateMetricsEngine() {
    if (this.engineMode === 'demo') {
      this.metrics.engine = 'Simulated Engine';
    } else if (this.engineMode === 'bhashini') {
      this.metrics.engine = 'Bhashini Cloud API';
    } else if (this.engineMode === 'cloud_llm') {
      this.metrics.engine = `Cloud LLM (${this.selectedLlmModel})`;
    } else {
      this.metrics.engine = 'AI4Bharat Local API';
    }
  }

  checkBackendStatus() {
    this.http.get<any>(`${this.cleanBackendBaseUrl}/status`)
      .pipe(
        catchError((err) => {
          this.serverStatus = 'offline';
          this.serverStatusText = 'Disconnected';
          this.deviceTarget = 'CPU';
          this.modelStates.asr = 'not_loaded';
          this.modelStates.en_indic = 'not_loaded';
          this.modelStates.indic_en = 'not_loaded';
          this.modelStates.tts = 'not_loaded';
          this.areLocalModelsLoaded = false;
          throw err;
        })
      )
      .subscribe((res) => {
        if (res.loading) {
          this.serverStatus = 'loading';
          this.serverStatusText = 'Loading Models';
          this.isBackendLoading = true;
        } else {
          this.serverStatus = 'online';
          this.serverStatusText = 'Connected';
          this.isBackendLoading = false;
        }
        if (res.configured_models) {
          this.configuredModels = res.configured_models;
        }
        this.deviceTarget = res.device.toUpperCase();
        this.modelStates.asr = res.models.indic_conformer_asr;
        this.modelStates.en_indic = res.models.indictrans2_en_indic;
        this.modelStates.indic_en = res.models.indictrans2_indic_en;
        this.modelStates.tts = res.models.indicf5_tts || 'not_loaded';

        this.areLocalModelsLoaded =
          this.modelStates.asr === 'ready' &&
          this.modelStates.en_indic === 'ready' &&
          this.modelStates.indic_en === 'ready' &&
          (this.modelStates.tts === 'ready' || this.modelStates.tts === 'ready_mock');

        // System metrics
        this.sysMetrics.cpuUsage = `${res.system.cpu_usage_percent}%`;
        const totalRam = res.system.memory_total_gb;
        const usedRamPercent = res.system.memory_usage_percent;
        const usedRam = ((totalRam * usedRamPercent) / 100).toFixed(1);
        this.sysMetrics.ramUsage = `${usedRam} GB / ${totalRam.toFixed(1)} GB`;
      });
  }

  triggerLoadBackendModels() {
    this.isBackendLoading = true;
    this.serverStatus = 'loading';
    this.serverStatusText = 'Downloading/Loading...';

    this.http.post(`${this.cleanBackendBaseUrl}/load`, { hf_token: this.hfToken }).subscribe({
      next: () => {
        this.checkBackendStatus();
      },
      error: (err) => {
        console.error("Failed to trigger model load:", err);
        this.isBackendLoading = false;
      }
    });
  }

  swapLanguages() {
    const temp = this.sourceLang;
    this.sourceLang = this.targetLang;
    this.targetLang = temp;

    const tempText = this.sourceText;
    this.sourceText = this.targetText;
    this.targetText = tempText;
  }

  clearSource() {
    this.sourceText = '';
    this.targetText = '';
  }

  onLanguageChange() {
    // Empty output if input translation lang changes
    this.targetText = '';
  }

  onSourceTextChange() {
    // If user clears manually
    if (!this.sourceText.trim()) {
      this.targetText = '';
    }
  }

  performTranslation() {
    if (!this.sourceText.trim()) return;
    this.isTranslating = true;

    const startTime = Date.now();

    if (this.engineMode === 'demo') {
      setTimeout(() => {
        this.targetText = this.getMockTranslationText(this.sourceText, this.sourceLang, this.targetLang);
        const latency = Date.now() - startTime;
        
        // Mock token metrics for demo mode
        const mockInputTokens = Math.ceil(this.sourceText.length / 4);
        const mockOutputTokens = Math.ceil(this.targetText.length / 4);
        const mockCost = Math.round((mockInputTokens * 0.15 + mockOutputTokens * 0.60) / 10000 * this.usdToInrRate * 10000) / 10000;
        
        this.updateTelemetryMetrics(latency, this.sourceText.length, 'Simulated Engine', mockCost, mockInputTokens, mockOutputTokens);
        this.logHistory(this.sourceText, this.targetText, latency, 'Simulated Engine', mockCost, mockInputTokens, mockOutputTokens);
        this.isTranslating = false;
      }, 300);

    } else if (this.engineMode === 'local' || this.engineMode === 'cloud_llm') {
      const payload: any = {
        text: this.sourceText,
        src_lang: this.sourceLang,
        tgt_lang: this.targetLang,
        engine_mode: this.engineMode
      };

      if (this.engineMode === 'cloud_llm') {
        payload.model = this.selectedLlmModel;
        payload.api_keys = this.llmKeys;
        payload.usd_to_inr_rate = this.usdToInrRate;
      }

      this.http.post<any>(`${this.cleanBackendBaseUrl}/translate`, payload).subscribe({
        next: (res) => {
          this.targetText = res.translation || res.translated_text;
          const latency = Date.now() - startTime;
          const tokenMetrics = res.token_metrics || { cost_inr: 0, input_tokens: 0, output_tokens: 0 };
          this.updateTelemetryMetrics(
            res.metrics.latency_ms || latency,
            this.sourceText.length,
            res.metrics.engine,
            tokenMetrics.cost_inr,
            tokenMetrics.input_tokens,
            tokenMetrics.output_tokens
          );
          this.logHistory(
            this.sourceText,
            this.targetText,
            res.metrics.latency_ms || latency,
            res.metrics.engine,
            tokenMetrics.cost_inr,
            tokenMetrics.input_tokens,
            tokenMetrics.output_tokens
          );
          this.isTranslating = false;
        },
        error: (err) => {
          console.error("Translation failed:", err);
          const errorMsg = err.error?.detail || err.message || "Ensure the python backend server is running.";
          alert("Translation failed: " + errorMsg);
          this.isTranslating = false;
        }
      });

    } else if (this.engineMode === 'bhashini') {
      if (!this.bhashiniConfig.userId || !this.bhashiniConfig.apiKey || !this.bhashiniConfig.authToken) {
        alert("Please provide User ID, API Key, and Auth Token in the Setup panel.");
        this.isTranslating = false;
        return;
      }

      this.translateViaBhashini(startTime);
    }
  }

  translateViaBhashini(startTime: number) {
    const url = 'https://dhruva-api.bhashini.gov.in/services/inference/pipeline';
    const headers = {
      'ulcaApiKey': this.bhashiniConfig.apiKey,
      'userID': this.bhashiniConfig.userId,
      'Authorization': this.bhashiniConfig.authToken,
      'Content-Type': 'application/json'
    };

    const payload = {
      pipelineTasks: [
        {
          taskType: 'translation',
          config: {
            language: {
              sourceLanguage: this.sourceLang,
              targetLanguage: this.targetLang
            },
            serviceId: 'ai4bharat/indictrans-v2-all-gpu--t4'
          }
        }
      ],
      inputData: {
        input: [{ source: this.sourceText }]
      }
    };

    this.http.post<any>(url, payload, { headers }).subscribe({
      next: (res) => {
        try {
          const translated = res.pipelineResponse[0].output[0].target;
          this.targetText = translated;
          const latency = Date.now() - startTime;
          
          // Calculate character count and Bhashini costing
          const inputChars = this.sourceText.length;
          const outputChars = translated.length;
          const bhashiniCost = (inputChars * this.bhashiniRates.translationRatePer1000) / 1000;
          
          this.updateTelemetryMetrics(latency, this.sourceText.length, 'Bhashini Cloud API', bhashiniCost, inputChars, outputChars);
          this.logHistory(this.sourceText, this.targetText, latency, 'Bhashini Cloud API', bhashiniCost, inputChars, outputChars);
        } catch (e) {
          console.error("Failed to parse Bhashini response", e);
          alert("Bhashini API key rejected or serviceId configuration error.");
        }
        this.isTranslating = false;
      },
      error: (err) => {
        console.error("Bhashini cloud translation failed:", err);
        alert(`Bhashini API Error: ${err.message || 'Verification failed. Confirm credentials.'}`);
        this.isTranslating = false;
      }
    });
  }

  updateTelemetryMetrics(
    latencyMs: number,
    charsCount: number,
    engine: string,
    costInr: number = 0,
    inputTokens: number = 0,
    outputTokens: number = 0
  ) {
    this.metrics.latency = latencyMs;
    this.metrics.cps = latencyMs > 0 ? Math.round((charsCount / (latencyMs / 1000.0)) * 100) / 100 : charsCount;
    this.metrics.engine = engine;
    this.metrics.costInr = costInr;
    this.metrics.inputTokens = inputTokens;
    this.metrics.outputTokens = outputTokens;
  }

  logHistory(
    srcText: string,
    tgtText: string,
    latency: number,
    engine: string,
    costInr: number = 0,
    inputTokens: number = 0,
    outputTokens: number = 0
  ) {
    const entry: HistoryEntry = {
      timestamp: new Date(),
      srcLang: this.sourceLang,
      tgtLang: this.targetLang,
      srcText,
      tgtText,
      latency,
      cps: latency > 0 ? Math.round((srcText.length / (latency / 1000.0)) * 100) / 100 : srcText.length,
      engine,
      costInr,
      inputTokens,
      outputTokens
    };

    // Add to top of list
    this.history.unshift(entry);

    // Cap history length at 50
    if (this.history.length > 50) {
      this.history.pop();
    }

    localStorage.setItem('mitra_translation_history', JSON.stringify(this.history));
  }

  clearHistory() {
    this.history = [];
    localStorage.removeItem('mitra_translation_history');
  }

  exportHistory(type: 'json' | 'csv') {
    let dataStr = '';
    let mimeType = '';
    let filename = '';

    if (type === 'json') {
      dataStr = JSON.stringify(this.history, null, 2);
      mimeType = 'application/json';
      filename = 'mitra_efficiency_metrics.json';
    } else {
      // CSV Export
      const headers = ['Timestamp', 'Source Language', 'Target Language', 'Source Text', 'Translation', 'Latency (ms)', 'Throughput (char/s)', 'Engine', 'Input Tokens', 'Output Tokens', 'Total Tokens', 'Cost (INR)'];
      const rows = this.history.map(e => [
        new Date(e.timestamp).toISOString(),
        e.srcLang,
        e.tgtLang,
        `"${e.srcText.replace(/"/g, '""')}"`,
        `"${e.tgtText.replace(/"/g, '""')}"`,
        e.latency,
        e.cps,
        e.engine,
        e.inputTokens || 0,
        e.outputTokens || 0,
        (e.inputTokens || 0) + (e.outputTokens || 0),
        e.costInr ? `₹${e.costInr.toFixed(4)}` : '₹0.0000'
      ]);
      dataStr = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
      mimeType = 'text/csv';
      filename = 'mitra_efficiency_metrics.csv';
    }

    const blob = new Blob([dataStr], { type: mimeType });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.click();
    window.URL.revokeObjectURL(url);
  }

  // Voice Speech Recording Control
  toggleRecording() {
    if (this.isRecording) {
      this.stopRecording();
    } else {
      this.startRecording();
    }
  }

  startRecording() {
    if (this.engineMode === 'demo') {
      this.isRecording = true;
      this.metrics.asrLatency = 0;

      // Simulate Voice recording translation workflow
      setTimeout(() => {
        this.isRecording = false;
        const mockPhrases: Record<string, string> = {
          'hi': 'नमस्ते, आप कैसे हैं?',
          'ta': 'வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?',
          'te': 'నమస్కారం, మీరు ఎలా ఉన్నారు?',
          'kn': 'ನಮಸ್ಕಾರ, ನೀವು ಹೇಗಿದ್ದೀರಾ?',
          'ml': 'നമസ്കാരം, സുഖമാണോ?',
          'bn': 'হ্যালো, আপনি কেমন আছেন?',
          'mr': 'नमस्कार, तुम्ही कसे आहात?',
          'gu': 'નમસ્તે, તમે કેમ છો?',
          'en': 'Hello, how are you today?'
        };
        this.sourceText = mockPhrases[this.sourceLang] || 'Hello, testing simulated speech to text.';
        this.metrics.asrLatency = 450; // Mock decoding speed
        this.performTranslation();
      }, 2000);

      return;
    }

    // Actual Audio Recording via WavRecorder
    this.wavRecorder = new WavRecorder();
    this.wavRecorder.start()
      .then(() => {
        this.isRecording = true;
      })
      .catch((err) => {
        console.error("Microphone recording failed:", err);
        alert("Microphone permission denied or Web Audio unsupported in this browser.");
      });
  }

  stopRecording() {
    if (this.wavRecorder && this.isRecording) {
      const audioBlob = this.wavRecorder.stop();
      this.isRecording = false;
      this.uploadAndTranscribeAudio(audioBlob);
    }
  }

  uploadAndTranscribeAudio(audioBlob: Blob) {
    this.isTranslating = true;
    const startTime = Date.now();

    const formData = new FormData();
    // Send as input.wav
    formData.append('file', audioBlob, 'input.wav');
    formData.append('lang', this.sourceLang);

    this.http.post<any>(`${this.cleanBackendBaseUrl}/transcribe`, formData).subscribe({
      next: (res) => {
        this.sourceText = res.text;
        this.metrics.asrLatency = res.metrics.latency_ms || (Date.now() - startTime);

        // Follow up with translation automatically
        this.performTranslation();
      },
      error: (err) => {
        console.error("Voice transcription failed:", err);
        alert("Failed to transcribe audio. Check that backend server is active and model is loaded.");
        this.isTranslating = false;
      }
    });
  }

  onRefAudioChange(event: any) {
    const file = event.target.files[0];
    if (file) {
      if (file.type !== 'audio/wav' && !file.name.endsWith('.wav')) {
        alert("Please upload a .wav audio file.");
        event.target.value = '';
        return;
      }
      this.refAudioFile = file;
    }
  }

  // Text to Speech Translation Playback
  listenSourceText() {
    this.activeTtsSide = 'source';
    this.listenText(this.sourceText, this.sourceLang);
  }

  listenTargetText() {
    this.activeTtsSide = 'target';
    this.listenText(this.targetText, this.targetLang);
  }

  listenText(text: string, langCode: string) {
    if (!text.trim()) return;

    // Fall back to native browser speech synthesis for English, as IndicF5 only supports Indian languages
    if (langCode === 'en') {
      if ('speechSynthesis' in window) {
        if (this.currentAudio) {
          this.currentAudio.pause();
          this.currentAudio = null;
        }
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = 'en-US';
        window.speechSynthesis.speak(utterance);
      } else {
        alert("Web Speech Synthesis is not supported in this browser.");
      }
      return;
    }

    // Local IndicF5 TTS model (via backend /synthesize)
    this.isSynthesizing = true;
    const startTime = Date.now();

    const formData = new FormData();
    formData.append('text', text);

    if (this.ttsVoiceMode === 'clone' && this.refAudioFile && this.refTranscript) {
      formData.append('ref_audio', this.refAudioFile, this.refAudioFile.name);
      formData.append('ref_text', this.refTranscript);
    }

    this.http.post(`${this.cleanBackendBaseUrl}/synthesize`, formData, { responseType: 'blob' }).subscribe({
      next: (blob: Blob) => {
        this.synthesisLatency = Date.now() - startTime;
        this.isSynthesizing = false;
        this.activeTtsSide = null;

        if (this.currentAudio) {
          this.currentAudio.pause();
          this.currentAudio = null;
        }

        const audioUrl = URL.createObjectURL(blob);
        this.currentAudio = new Audio(audioUrl);
        this.currentAudio.play().catch(e => {
          console.error("Audio playback failed:", e);
          alert("Audio playback failed. Please check your browser audio configurations.");
        });
      },
      error: (err) => {
        console.error("TTS synthesis failed:", err);
        alert("TTS synthesis failed. Make sure the local models are loaded and backend is running.");
        this.isSynthesizing = false;
        this.activeTtsSide = null;
      }
    });
  }

  copyTargetToClipboard() {
    if (!this.targetText.trim()) return;

    navigator.clipboard.writeText(this.targetText).then(() => {
      this.showCopyToast = true;
      setTimeout(() => {
        this.showCopyToast = false;
      }, 2000);
    });
  }

  // Pre-packaged translation samples for Demo/Mock Mode to show visual feedback instantly
  private getMockTranslationText(text: string, src: string, tgt: string): string {
    const cleaned = text.trim().toLowerCase().replace(/[?.,!]/g, '');

    const mockDb: Record<string, Record<string, string>> = {
      'hello': { 'hi': 'नमस्ते', 'ta': 'வணக்கம்', 'te': 'నమస్కారం', 'kn': 'ನಮಸ್ಕಾರ', 'ml': 'നമസ്കാരം', 'bn': 'হ্যালো', 'mr': 'नमस्कार', 'gu': 'નમસ્તે', 'pa': 'ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ' },
      'how are you': { 'hi': 'आप कैसे हैं?', 'ta': 'நீங்கள் எப்படி இருக்கிறீர்கள்?', 'te': 'మీరు ఎలా ఉన్నారు?', 'kn': 'ನೀವು ಹೇಗಿದ್ದೀರಾ?', 'ml': 'സുഖമാണോ?', 'bn': 'আপনি কেমন আছেন?', 'mr': 'तुम्ही कसे आहात?', 'gu': 'તમે કેમ છો?', 'pa': 'ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ?' },
      'thank you': { 'hi': 'धन्यवाद', 'ta': 'நன்றி', 'te': 'ధన్యవాదాలు', 'kn': 'ಧನ್ಯವಾದಗಳು', 'ml': 'നന്ദി', 'bn': 'ধন্যবাদ', 'mr': 'धन्यवाद', 'gu': 'આભાર', 'pa': 'ਧੰਨਵਾਦ' },
      'good morning': { 'hi': 'शुभ प्रभात', 'ta': 'காலை வணக்கம்', 'te': 'శుభోదయం', 'kn': 'ಶುಭೋದಯ', 'ml': 'സുപ്രഭാതം', 'bn': 'সুপ্রভাত', 'mr': 'शुभ प्रभात', 'gu': 'શુભ સવાર', 'pa': 'ਸ਼ੁਭ ਸਵੇਰ' },
      'नमस्ते': { 'en': 'Hello / Greetings' },
      'आप कैसे हैं': { 'en': 'How are you?' },
      'धन्यवाद': { 'en': 'Thank you' },
      'शुभ प्रभात': { 'en': 'Good morning' },
      'வணக்கம்': { 'en': 'Hello / Greetings' },
      'நன்றி': { 'en': 'Thank you' }
    };

    // Look up in database
    if (mockDb[cleaned] && mockDb[cleaned][tgt]) {
      return mockDb[cleaned][tgt];
    }

    // Look up reversed
    if (src === 'en') {
      const langName = this.languages.find(l => l.code === tgt)?.name || tgt;
      return `[${langName} Translation of: "${text}"]`;
    } else {
      return `[English Translation of: "${text}"]`;
    }
  }
}

// Client-side WAV recording and downsampling helper
class WavRecorder {
  private audioContext: AudioContext | null = null;
  private mediaStream: MediaStream | null = null;
  private scriptProcessor: ScriptProcessorNode | null = null;
  private recordingBuffer: Float32Array[] = [];
  private sampleRate = 0;

  constructor() {}

  async start(): Promise<void> {
    this.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
    this.sampleRate = this.audioContext.sampleRate;

    const source = this.audioContext.createMediaStreamSource(this.mediaStream);
    this.scriptProcessor = this.audioContext.createScriptProcessor(4096, 1, 1);

    this.recordingBuffer = [];

    this.scriptProcessor.onaudioprocess = (event) => {
      const input = event.inputBuffer.getChannelData(0);
      this.recordingBuffer.push(new Float32Array(input));
    };

    source.connect(this.scriptProcessor);
    this.scriptProcessor.connect(this.audioContext.destination);
  }

  stop(): Blob {
    if (this.scriptProcessor && this.audioContext) {
      this.scriptProcessor.disconnect();
      this.audioContext.close();
    }
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(t => t.stop());
    }

    const totalLength = this.recordingBuffer.reduce((acc, val) => acc + val.length, 0);
    const result = new Float32Array(totalLength);
    let offset = 0;
    for (const buffer of this.recordingBuffer) {
      result.set(buffer, offset);
      offset += buffer.length;
    }

    const downsampled = this.downsample(result, this.sampleRate, 16000);
    return this.encodeWAV(downsampled, 16000);
  }

  private downsample(buffer: Float32Array, fromRate: number, toRate: number): Float32Array {
    if (fromRate === toRate) return buffer;
    const sampleRateRatio = fromRate / toRate;
    const newLength = Math.round(buffer.length / sampleRateRatio);
    const result = new Float32Array(newLength);
    let offsetResult = 0;
    let offsetBuffer = 0;
    while (offsetResult < result.length) {
      const nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
      let accum = 0, count = 0;
      for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
        accum += buffer[i];
        count++;
      }
      result[offsetResult] = count > 0 ? accum / count : 0;
      offsetResult++;
      offsetBuffer = nextOffsetBuffer;
    }
    return result;
  }

  private encodeWAV(samples: Float32Array, sampleRate: number): Blob {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    this.writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    this.writeString(view, 8, 'WAVE');
    this.writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    this.writeString(view, 36, 'data');
    view.setUint32(40, samples.length * 2, true);

    this.floatTo16BitPCM(view, 44, samples);

    return new Blob([view], { type: 'audio/wav' });
  }

  private floatTo16BitPCM(output: DataView, offset: number, input: Float32Array) {
    for (let i = 0; i < input.length; i++, offset += 2) {
      let s = Math.max(-1, Math.min(1, input[i]));
      output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
  }

  private writeString(view: DataView, offset: number, string: string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }
}
