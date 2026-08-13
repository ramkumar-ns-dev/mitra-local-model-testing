import time
import torch
from typing import Optional
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from backend.services.providers.base import BaseTranslationProvider
from backend.utils.logging_config import get_logger

logger = get_logger()


class IndicTrans2TranslationProvider(BaseTranslationProvider):
    """Provider for AI4Bharat IndicTrans2 models."""

    def __init__(self):
        self.model_id = None
        self.device = None
        self.model = None
        self.tokenizer = None
        self.processor = None

    def load(self, model_id: str, device: torch.device, hf_token: Optional[str] = None):
        self.model_id = model_id
        # Force CPU if MPS backend due to PyTorch float16 MPS op support limitations in IndicTrans2
        self.device = torch.device("cpu") if device.type == "mps" else device
        logger.info(f"Loading IndicTrans2 provider: '{model_id}' on {self.device}")
        
        from IndicTransToolkit.processor import IndicProcessor
        self.processor = IndicProcessor(inference=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            token=hf_token
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device.type != "cpu" else torch.float32,
            token=hf_token
        ).to(self.device)
        self.model.eval()
        logger.info(f"IndicTrans2 provider '{model_id}' loaded successfully on {self.device}.")

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        if not self.model or not self.tokenizer or not self.processor:
            raise RuntimeError(f"IndicTrans2 provider for '{self.model_id}' is not loaded.")

        batch = self.processor.preprocess_batch([text], src_lang=source_lang, tgt_lang=target_lang)
        inputs = self.tokenizer(batch, truncation=True, padding="longest", return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                num_beams=2,
                num_return_sequences=1,
                max_length=256
            )

        decoded = self.tokenizer.batch_decode(
            generated_tokens.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        translations = self.processor.postprocess_batch(decoded, lang=target_lang)
        return translations[0] if len(translations) > 0 else ""


class NLLBTranslationProvider(BaseTranslationProvider):
    """Provider for Meta NLLB translation models."""

    def __init__(self):
        self.model_id = None
        self.device = None
        self.pipeline = None

    def load(self, model_id: str, device: torch.device, hf_token: Optional[str] = None):
        self.model_id = model_id
        self.device = device
        logger.info(f"Loading NLLB translation provider: '{model_id}' on {device}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            token=hf_token
        )
        self.pipeline = pipeline(
            "translation",
            model=model,
            tokenizer=tokenizer,
            device=device
        )
        logger.info(f"NLLB provider '{model_id}' loaded successfully.")

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        if not self.pipeline:
            raise RuntimeError(f"NLLB translation provider for '{self.model_id}' is not loaded.")

        result = self.pipeline(text, src_lang=source_lang, tgt_lang=target_lang)
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("translation_text", "")
        elif isinstance(result, dict):
            return result.get("translation_text", "")
        return str(result)


class MockTranslationProvider(BaseTranslationProvider):
    """Mock Translation Provider for fast testing and offline mode."""

    def __init__(self):
        self.model_id = "mock-translation"

    def load(self, model_id: str, device: torch.device, hf_token: Optional[str] = None):
        self.model_id = model_id
        logger.info("Mock Translation provider ready.")

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        time.sleep(0.05)
        cleaned = text.strip().lower().rstrip("?.,!")
        if source_lang == "eng_Latn" and target_lang == "hin_Deva":
            db = {"hello": "नमस्ते", "how are you": "आप कैसे हैं", "hello how are you": "नमस्ते, आप कैसे हैं?"}
            return db.get(cleaned, f"नमस्ते, यह अनुवाद का अनुकरण है: '{text}'")
        if source_lang == "hin_Deva" and target_lang == "eng_Latn":
            db = {"नमस्ते": "Hello", "आप कैसे हैं": "How are you", "नमस्ते आप कैसे हैं": "Hello, how are you?"}
            return db.get(cleaned, f"Hello, simulated translation of: '{text}'")
        return f"[Translation from {source_lang} to {target_lang} of: '{text}']"
