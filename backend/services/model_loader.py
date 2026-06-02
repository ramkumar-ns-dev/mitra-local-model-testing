import time
import torch
import numpy as np
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from backend.config import settings
from backend.utils.logging_config import get_logger

logger = get_logger()

class ModelLoader:
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
        self.asr_pipeline = None
        self.translation_pipeline = None
        self.translation_tokenizer = None
        self.translation_model = None
        self.indic_processor = None
        self.asr_status = "not_loaded"
        self.translation_status = "not_loaded"
        self.asr_model_name = settings.ASR_MODEL_ID
        self.translation_model_name = settings.TRANSLATION_MODEL_ID
        self._initialized = True

    def _get_device(self) -> torch.device:
        """
        Determines the device to run inference on.
        """
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

    def load_models(self):
        """
        Loads ASR and translation models. Handles fallbacks and mock configurations.
        """
        if settings.MOCK_MODELS:
            logger.warning("MOCK_MODELS is enabled. Running in simulated mode without loading neural networks.")
            self.asr_status = "ready_mock"
            self.translation_status = "ready_mock"
            return

        # 1. Load Speech-to-Text (ASR) Model
        try:
            self.asr_status = "loading"
            logger.info(f"Loading ASR model: {self.asr_model_name}")
            
            # Since ai4bharat/conformer-hi-gpu--t4 is a Bhashini Service ID, it will fail to load from Hugging Face hub.
            # We explicitly raise an exception if it matches to trigger the fallback directly.
            if "gpu--t4" in self.asr_model_name:
                raise ValueError(f"{self.asr_model_name} is a Bhashini API service ID and cannot be loaded as a local HF repository.")
                
            self.asr_pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.asr_model_name,
                device=self.device
            )
            self.asr_status = "ready"
            logger.info("ASR model loaded successfully.")

        except Exception as e:
            logger.warning(f"Failed to load primary ASR model '{self.asr_model_name}': {e}. Attempting fallback to '{settings.FALLBACK_ASR_MODEL_ID}'...")
            try:
                self.asr_model_name = settings.FALLBACK_ASR_MODEL_ID
                self.asr_pipeline = pipeline(
                    "automatic-speech-recognition",
                    model=self.asr_model_name,
                    device=self.device
                )
                self.asr_status = "ready"
                logger.info(f"ASR fallback model '{settings.FALLBACK_ASR_MODEL_ID}' loaded successfully.")
            except Exception as fe:
                logger.error(f"Failed to load ASR fallback model: {fe}. Falling back to MOCK mode for ASR.")
                self.asr_status = "ready_mock"

        # 2. Load Translation Model
        try:
            self.translation_status = "loading"
            logger.info(f"Loading translation model: {self.translation_model_name}")
            
            if "indictrans2-en-indic" in self.translation_model_name.lower():
                from IndicTransToolkit.processor import IndicProcessor
                self.indic_processor = IndicProcessor(inference=True)
                
                self.translation_tokenizer = AutoTokenizer.from_pretrained(
                    self.translation_model_name,
                    trust_remote_code=True,
                    token=settings.HF_TOKEN
                )
                self.translation_model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.translation_model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                    token=settings.HF_TOKEN
                ).to(self.device)
                
                self.translation_model.eval()
                self.translation_status = "ready"
                logger.info("IndicTrans2 translation model loaded successfully.")
            else:
                tokenizer = AutoTokenizer.from_pretrained(self.translation_model_name, token=settings.HF_TOKEN)
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.translation_model_name,
                    torch_dtype=torch.float32,
                    token=settings.HF_TOKEN
                )
                
                self.translation_pipeline = pipeline(
                    "translation",
                    model=model,
                    tokenizer=tokenizer,
                    device=self.device
                )
                self.translation_status = "ready"
                logger.info("NLLB translation model loaded successfully.")

        except Exception as e:
            logger.error(f"Failed to load translation model '{self.translation_model_name}': {e}. Falling back to MOCK mode for translation.")
            self.translation_status = "ready_mock"

        # 3. Warm-up models
        self.warmup_models()

    def warmup_models(self):
        """
        Runs dummy inputs through the pipelines to warm up cache and reduce first-request latency.
        """
        logger.info("Warming up models...")
        
        # Warm-up ASR
        if self.asr_status == "ready" and self.asr_pipeline is not None:
            try:
                start = time.time()
                # 1-second of silence at 16kHz
                dummy_audio = np.zeros(16000, dtype=np.float32)
                self.asr_pipeline(dummy_audio)
                logger.info(f"ASR model warmed up in {time.time() - start:.2f} seconds.")
            except Exception as e:
                logger.error(f"ASR warm-up failed: {e}")
                
        # Warm-up Translation
        if self.translation_status == "ready":
            try:
                start = time.time()
                if "indictrans2-en-indic" in self.translation_model_name.lower():
                    self.translate("Hello", source_lang="eng_Latn", target_lang="hin_Deva")
                elif self.translation_pipeline is not None:
                    self.translation_pipeline(
                        "Hello", 
                        src_lang="eng_Latn", 
                        tgt_lang="hin_Deva"
                    )
                logger.info(f"Translation model warmed up in {time.time() - start:.2f} seconds.")
            except Exception as e:
                logger.error(f"Translation warm-up failed: {e}")

    def transcribe(self, waveform_tensor: torch.Tensor) -> dict:
        """
        Runs speech-to-text inference.
        """
        if self.asr_status == "ready_mock":
            # Simulate transcription
            time.sleep(0.5)  # Simulate network/processing latency
            return {"text": "नमस्ते आप कैसे हैं", "confidence": 0.98, "engine": "Mock ASR"}

        if not self.asr_pipeline:
            raise RuntimeError("ASR model is not initialized or loaded.")

        try:
            # Convert PyTorch tensor to numpy array (pipeline expectation)
            audio_array = waveform_tensor.numpy()
            
            # Perform speech recognition
            # Whisper and other models support language selection or auto-detect. 
            # We specify Hindi generation kwargs if the model supports it.
            gen_kwargs = {}
            if "whisper" in self.asr_model_name.lower():
                gen_kwargs = {"language": "hindi", "task": "transcribe"}

            result = self.asr_pipeline(audio_array, generate_kwargs=gen_kwargs)
            
            # Extract transcription and confidence
            text = result.get("text", "").strip()
            
            # Estimate a basic confidence score (some pipelines return chunks with logs)
            confidence = None
            if "chunks" in result:
                # If chunk-level probabilities are present, we can average them
                confidences = [c.get("confidence") for c in result["chunks"] if c.get("confidence") is not None]
                if confidences:
                    confidence = float(np.mean(confidences))
                    
            return {
                "text": text if text else "[Unintelligible speech]",
                "confidence": confidence,
                "engine": f"Local ASR ({self.asr_model_name})"
            }

        except Exception as e:
            logger.error(f"Inference error during ASR transcription: {e}")
            raise RuntimeError(f"ASR inference failed: {str(e)}")

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translates text from source_lang to target_lang.
        """
        if self.translation_status == "ready_mock":
            time.sleep(0.2)  # Simulate processing latency
            return self._get_mock_translation(text, source_lang, target_lang)

        # 1. Handle IndicTrans2 Translation Sequence
        if "indictrans2-en-indic" in self.translation_model_name.lower():
            if not self.translation_model or not self.translation_tokenizer or not self.indic_processor:
                raise RuntimeError("IndicTrans2 translation model is not fully initialized.")
            
            try:
                # Preprocess
                batch = self.indic_processor.preprocess_batch([text], src_lang=source_lang, tgt_lang=target_lang)
                # Tokenize
                inputs = self.translation_tokenizer(batch, truncation=True, padding="longest", return_tensors="pt").to(self.device)
                # Generate
                with torch.no_grad():
                    generated_tokens = self.translation_model.generate(
                        **inputs,
                        num_beams=5,
                        num_return_sequences=1,
                        max_length=256
                    )
                # Decode
                decoded = self.translation_tokenizer.batch_decode(
                    generated_tokens.detach().cpu().tolist(),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                # Postprocess
                translations = self.indic_processor.postprocess_batch(decoded, lang=target_lang)
                return translations[0] if len(translations) > 0 else ""
            except Exception as e:
                logger.error(f"Inference error during IndicTrans2 translation: {e}")
                raise RuntimeError(f"Translation inference failed: {str(e)}")

        # 2. Handle standard NLLB Pipeline Translation
        if not self.translation_pipeline:
            raise RuntimeError("Translation model is not initialized or loaded.")

        try:
            result = self.translation_pipeline(
                text,
                src_lang=source_lang,
                tgt_lang=target_lang
            )
            
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("translation_text", "")
            elif isinstance(result, dict):
                return result.get("translation_text", "")
            return str(result)

        except Exception as e:
            logger.error(f"Inference error during translation: {e}")
            raise RuntimeError(f"Translation inference failed: {str(e)}")

    def _get_mock_translation(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Pre-packaged mock translations for common inputs to support instant offline validation.
        """
        # Clean text
        cleaned = text.strip().lower().rstrip("?.,!")
        
        # English to Hindi Devanagari common lookups
        if source_lang == "eng_Latn" and target_lang == "hin_Deva":
            db = {
                "hello, how are you": "नमस्ते, आप कैसे हैं?",
                "hello how are you": "नमस्ते, आप कैसे हैं?",
                "how are you": "आप कैसे हैं?",
                "what is your name": "आपका नाम क्या है?",
                "thank you": "धन्यवाद",
                "good morning": "शुभ प्रभात",
                "hello": "नमस्ते"
            }
            if cleaned in db:
                return db[cleaned]
            return f"नमस्ते, यह अनुवाद का अनुकरण है: '{text}'"
            
        # Hindi Devanagari to English common lookups
        if source_lang == "hin_Deva" and target_lang == "eng_Latn":
            db = {
                "नमस्ते": "Hello",
                "आप कैसे हैं": "How are you?",
                "नमस्ते आप कैसे हैं": "Hello, how are you?",
                "धन्यवाद": "Thank you",
                "शुभ प्रभात": "Good morning"
            }
            if cleaned in db:
                return db[cleaned]
            return f"Hello, this is a simulated translation of: '{text}'"

        return f"[Translation from {source_lang} to {target_lang} of: '{text}']"

    def get_status(self) -> dict:
        """
        Returns model loading status and devices.
        """
        return {
            "asr_status": self.asr_status,
            "asr_model": self.asr_model_name,
            "translation_status": self.translation_status,
            "translation_model": self.translation_model_name,
            "device": str(self.device)
        }

# Global singleton
model_loader = ModelLoader()
