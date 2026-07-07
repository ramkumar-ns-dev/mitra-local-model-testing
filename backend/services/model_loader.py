import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import time
import sys
import torch
import numpy as np
import threading

# Monkey-patch torch.compile to prevent Dynamo Python 3.12+ compatibility errors in IndicF5
if sys.version_info >= (3, 12):
    torch.compile = lambda model, *args, **kwargs: model

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
        self.tts_status = "not_loaded"
        self.asr_model_name = settings.ASR_MODEL_ID
        self.translation_model_name = settings.TRANSLATION_MODEL_ID
        self.tts_model_name = settings.TTS_MODEL_ID
        self.tts_model = None
        self.tts_lock = threading.Lock()
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
            
            if "indic-conformer" in self.asr_model_name.lower():
                from transformers import AutoModel
                self.asr_model = AutoModel.from_pretrained(
                    self.asr_model_name,
                    trust_remote_code=True,
                    token=settings.HF_TOKEN
                ).to(self.device)
                self.asr_status = "ready"
                logger.info("IndicConformer ASR model loaded successfully.")
            else:
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
            
            if "indictrans2" in self.translation_model_name.lower():
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
                    torch_dtype=torch.float16 if self.device.type != "cpu" else torch.float32,
                    token=settings.HF_TOKEN
                ).to(self.device)
                
                self.translation_model.eval()
                self.translation_status = "ready"
                logger.info(f"IndicTrans2 translation model '{self.translation_model_name}' loaded successfully.")
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

        # 3. Load Text-to-Speech (TTS) Model
        if settings.MOCK_MODELS:
            self.tts_status = "ready_mock"
        else:
            try:
                self.tts_status = "loading"
                logger.info(f"Loading TTS model: {self.tts_model_name}")
                from transformers import AutoModel
                from safetensors.torch import load_file
                from huggingface_hub import hf_hub_download

                # Force CPU for TTS model if device is MPS due to lack of ComplexFloat support on MPS backend in PyTorch
                self.tts_device = torch.device("cpu") if self.device.type == "mps" else self.device
                logger.info(f"Loading TTS model structure on device: {self.tts_device}")
                self.tts_model = AutoModel.from_pretrained(
                    self.tts_model_name,
                    trust_remote_code=True,
                    token=settings.HF_TOKEN
                ).to(self.tts_device)
                
                logger.info("Loading safetensors checkpoint and cleaning state dict keys...")
                safetensors_path = hf_hub_download(self.tts_model_name, filename="model.safetensors", token=settings.HF_TOKEN)
                state_dict = load_file(safetensors_path, device="cpu")
                cleaned_state_dict = {}
                for k, v in state_dict.items():
                    new_key = k.replace("._orig_mod.", ".")
                    cleaned_state_dict[new_key] = v
                    
                self.tts_model.load_state_dict(cleaned_state_dict, strict=False)
                self.tts_model.config.remove_sil = False


                # Monkey-patch default nfe_step from 32 to 16 for faster CPU inference
                try:
                    import f5_tts.infer.utils_infer
                    f5_tts.infer.utils_infer.nfe_step = 16
                    logger.info("Monkey-patched f5_tts.infer.utils_infer.nfe_step to 16")
                except Exception as monkey_err:
                    logger.warning(f"Failed to monkey patch nfe_step: {monkey_err}")
                
                self.tts_status = "ready"
                logger.info(f"IndicF5 TTS model loaded successfully on {self.tts_device}.")
            except Exception as e:
                logger.error(f"Failed to load TTS model '{self.tts_model_name}': {e}. Falling back to MOCK mode for TTS.")
                self.tts_status = "ready_mock"

        # 4. Warm-up models (commented out to prevent blocking server startup)
        # self.warmup_models()

    def warmup_models(self):
        """
        Runs dummy inputs through the pipelines to warm up cache and reduce first-request latency.
        """
        logger.info("Warming up models...")
        
        # Warm-up ASR
        if self.asr_status == "ready":
            if "indic-conformer" in self.asr_model_name.lower() and hasattr(self, "asr_model"):
                try:
                    start = time.time()
                    dummy_audio = torch.zeros((1, 16000), dtype=torch.float32).to(self.device)
                    self.asr_model(dummy_audio, lang="hi", decoding="ctc")
                    logger.info(f"ASR model warmed up in {time.time() - start:.2f} seconds.")
                except Exception as e:
                    logger.error(f"ASR warm-up failed: {e}")
            elif self.asr_pipeline is not None:
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

        # Warm-up TTS
        if self.tts_status == "ready" and hasattr(self, "tts_model") and self.tts_model is not None:
            try:
                start = time.time()
                self.tts_model(
                    "नमस्ते", 
                    ref_audio_path=settings.DEFAULT_REF_AUDIO_PATH, 
                    ref_text=settings.DEFAULT_REF_TEXT
                )
                logger.info(f"TTS model warmed up in {time.time() - start:.2f} seconds.")
            except Exception as e:
                logger.error(f"TTS warm-up failed: {e}")

    def transcribe(self, waveform_tensor: torch.Tensor) -> dict:
        """
        Runs speech-to-text inference.
        """
        if self.asr_status == "ready_mock":
            # Simulate transcription
            time.sleep(0.5)  # Simulate network/processing latency
            return {"text": "नमस्ते आप कैसे हैं", "confidence": 0.98, "engine": "Mock ASR"}

        if "indic-conformer" in self.asr_model_name.lower():
            if not hasattr(self, "asr_model") or self.asr_model is None:
                raise RuntimeError("IndicConformer model is not initialized or loaded.")
            try:
                # Prepare 2D tensor on target device
                wav_2d = waveform_tensor.unsqueeze(0).to(self.device)
                
                # Run inference
                with torch.no_grad():
                    text = self.asr_model(wav_2d, lang="hi", decoding="ctc")
                    
                return {
                    "text": text if text else "[Unintelligible speech]",
                    "confidence": None,
                    "engine": f"Local ASR ({self.asr_model_name})"
                }
            except Exception as e:
                logger.error(f"Inference error during IndicConformer transcription: {e}")
                raise RuntimeError(f"IndicConformer inference failed: {str(e)}")

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
        if "indictrans2" in self.translation_model_name.lower():
            if not self.translation_model or not self.translation_tokenizer or not self.indic_processor:
                raise RuntimeError("IndicTrans2 translation model is not fully initialized.")
            
            try:
                # Preprocess
                batch = self.indic_processor.preprocess_batch([text], src_lang=source_lang, tgt_lang=target_lang)
                # Tokenize
                inputs = self.translation_tokenizer(batch, truncation=True, padding="longest", return_tensors="pt").to(self.device)
                
                # Reduce num_beams from 5 to 2 to prevent 504 timeouts
                num_beams = 2
                
                # Generate
                with torch.no_grad():
                    generated_tokens = self.translation_model.generate(
                        **inputs,
                        num_beams=num_beams,
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
            "tts_status": self.tts_status,
            "tts_model": self.tts_model_name,
            "device": str(self.device)
        }

    def synthesize(self, text: str, ref_audio_bytes: bytes = None, ref_text: str = None) -> bytes:
        """
        Synthesizes speech from target text using IndicF5.
        If ref_audio_bytes/ref_text is not provided, falls back to the default reference voice.
        Returns WAV bytes (24kHz).
        """
        if self.tts_status == "ready_mock":
            import io
            import soundfile as sf
            logger.info("Mock synthesis requested.")
            # 1 second of silence at 24000Hz
            silent_audio = np.zeros(24000, dtype=np.float32)
            out_buf = io.BytesIO()
            sf.write(out_buf, silent_audio, 24000, format="WAV")
            return out_buf.getvalue()

        if not self.tts_model:
            raise RuntimeError("TTS model is not initialized or loaded.")

        with self.tts_lock:
            import io
            import tempfile
            import soundfile as sf
            import os

            temp_ref_path = None
            try:
                # Determine reference audio and transcript
                if ref_audio_bytes:
                    # Save uploaded bytes to a temp file
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_ref:
                        temp_ref.write(ref_audio_bytes)
                        temp_ref_path = temp_ref.name
                    transcript = ref_text or ""
                    logger.info(f"Using custom voice reference audio: path={temp_ref_path}, transcript_len={len(transcript)}")
                else:
                    temp_ref_path = settings.DEFAULT_REF_AUDIO_PATH
                    transcript = settings.DEFAULT_REF_TEXT
                    logger.info("Using default fallback voice reference audio")

                # Clean input text and transcript of punctuation to prevent silence generation in IndicF5
                import re
                cleaned_text = re.sub(r'[।॥.,?!;:\"\'\-\(\)\[\]\{\}]', ' ', text)
                cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
                
                cleaned_transcript = re.sub(r'[।॥.,?!;:\"\'\-\(\)\[\]\{\}]', ' ', transcript)
                cleaned_transcript = re.sub(r'\s+', ' ', cleaned_transcript).strip()

                # Run inference
                logger.info(f"Running IndicF5 synthesis for cleaned text: '{cleaned_text}' (original len: {len(text)})")
                start_time = time.time()
                with torch.no_grad():
                    audio_array = self.tts_model(
                        cleaned_text,
                        ref_audio_path=temp_ref_path,
                        ref_text=cleaned_transcript
                    )
                logger.info(f"IndicF5 model forward pass completed in {time.time() - start_time:.2f} seconds")

                # Normalize and convert audio to WAV bytes at 24kHz
                if audio_array.dtype == np.int16:
                    audio_array = audio_array.astype(np.float32) / 32768.0

                out_buf = io.BytesIO()
                sf.write(out_buf, np.array(audio_array, dtype=np.float32), 24000, format="WAV")
                return out_buf.getvalue()

            except Exception as e:
                logger.error(f"Inference error during IndicF5 synthesis: {e}")
                raise RuntimeError(f"IndicF5 synthesis failed: {str(e)}")
            finally:
                # Cleanup temp custom audio file if created
                if ref_audio_bytes and temp_ref_path and os.path.exists(temp_ref_path):
                    try:
                        os.remove(temp_ref_path)
                    except Exception as cleanup_err:
                        logger.warning(f"Failed to clean up temp reference file: {cleanup_err}")

# Global singleton
model_loader = ModelLoader()
