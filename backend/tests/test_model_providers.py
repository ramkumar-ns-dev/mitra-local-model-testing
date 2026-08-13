import unittest
import torch

from backend.services.providers.factory import ASRFactory, TranslationFactory, TTSFactory
from backend.services.providers.asr_providers import (
    WhisperASRProvider, IndicConformerASRProvider, MockASRProvider, WHISPER_LANG_MAP
)
from backend.services.providers.translation_providers import (
    IndicTrans2TranslationProvider, NLLBTranslationProvider, MockTranslationProvider
)
from backend.services.providers.tts_providers import (
    IndicF5TTSProvider, MockTTSProvider
)
from backend.services.providers.base import BaseASRProvider
from backend.services.model_loader import ModelLoader


class CustomTamilWhisperProvider(BaseASRProvider):
    """Example custom model provider for future models."""
    def load(self, model_id: str, device: torch.device, hf_token=None):
        pass

    def transcribe(self, waveform_tensor: torch.Tensor, language: str = "ta") -> dict:
        return {"text": "வணக்கம் தமிழ்", "confidence": 0.99, "engine": "Custom Tamil Whisper"}


class TestModelProvidersAndFactory(unittest.TestCase):

    def test_asr_factory_matching(self):
        """Test ASR factory resolution for various model identifiers."""
        # 1. IndicConformer model
        conformer_provider = ASRFactory.get_provider("ai4bharat/indic-conformer-600m-multilingual")
        self.assertIsInstance(conformer_provider, IndicConformerASRProvider)

        # 2. Whisper model (standard OpenAI)
        whisper_provider = ASRFactory.get_provider("openai/whisper-tiny")
        self.assertIsInstance(whisper_provider, WhisperASRProvider)

        # 3. Whisper Tamil model (fine-tuned model from HuggingFace)
        tamil_whisper_provider = ASRFactory.get_provider("vasista22/whisper-tamil-small")
        self.assertIsInstance(tamil_whisper_provider, WhisperASRProvider)

        # 4. Unknown model defaults to Whisper provider
        unknown_provider = ASRFactory.get_provider("some-random/speech-model")
        self.assertIsInstance(unknown_provider, WhisperASRProvider)

        # 5. Mock mode
        mock_provider = ASRFactory.get_provider("openai/whisper-tiny", is_mock=True)
        self.assertIsInstance(mock_provider, MockASRProvider)

    def test_dynamic_provider_registration(self):
        """Test registering a custom new model family dynamically into ASRFactory."""
        ASRFactory.register_provider("custom-tamil", CustomTamilWhisperProvider)
        provider = ASRFactory.get_provider("my-repo/custom-tamil-model-v1")
        self.assertIsInstance(provider, CustomTamilWhisperProvider)

        result = provider.transcribe(torch.zeros(16000))
        self.assertEqual(result["text"], "வணக்கம் தமிழ்")
        self.assertEqual(result["engine"], "Custom Tamil Whisper")

    def test_translation_factory_matching(self):
        """Test Translation factory resolution for various model identifiers."""
        # 1. IndicTrans2
        indic_provider = TranslationFactory.get_provider("ai4bharat/indictrans2-en-indic-1B")
        self.assertIsInstance(indic_provider, IndicTrans2TranslationProvider)

        # 2. NLLB
        nllb_provider = TranslationFactory.get_provider("facebook/nllb-200-distilled-600M")
        self.assertIsInstance(nllb_provider, NLLBTranslationProvider)

        # 3. Mock mode
        mock_provider = TranslationFactory.get_provider("nllb", is_mock=True)
        self.assertIsInstance(mock_provider, MockTranslationProvider)

    def test_tts_factory_matching(self):
        """Test TTS factory resolution for various model identifiers."""
        indicf5_provider = TTSFactory.get_provider("ai4bharat/IndicF5")
        self.assertIsInstance(indicf5_provider, IndicF5TTSProvider)

        mock_provider = TTSFactory.get_provider("IndicF5", is_mock=True)
        self.assertIsInstance(mock_provider, MockTTSProvider)

    def test_whisper_language_code_mapping(self):
        """Verify ISO and Bhashini language code mapping for Whisper."""
        self.assertEqual(WHISPER_LANG_MAP.get("ta"), "tamil")
        self.assertEqual(WHISPER_LANG_MAP.get("tam_Taml"), "tamil")
        self.assertEqual(WHISPER_LANG_MAP.get("hi"), "hindi")
        self.assertEqual(WHISPER_LANG_MAP.get("te"), "telugu")
        self.assertEqual(WHISPER_LANG_MAP.get("kn"), "kannada")

    def test_mock_asr_provider_multi_language(self):
        """Test Mock ASR provider returns correct language text simulated outputs."""
        provider = MockASRProvider()
        dummy_audio = torch.zeros(16000)

        res_ta = provider.transcribe(dummy_audio, language="ta")
        self.assertEqual(res_ta["text"], "வணக்கம் நீங்கள் எப்படி இருக்கிறீர்கள்")

        res_hi = provider.transcribe(dummy_audio, language="hi")
        self.assertEqual(res_hi["text"], "नमस्ते आप कैसे हैं")

    def test_model_loader_mock_orchestration(self):
        """Test ModelLoader singleton orchestration in Mock mode."""
        loader = ModelLoader()
        loader.asr_provider = ASRFactory.get_provider("mock", is_mock=True)
        loader.translation_provider = TranslationFactory.get_provider("mock", is_mock=True)
        loader.tts_provider = TTSFactory.get_provider("mock", is_mock=True)

        loader.asr_status = "ready_mock"
        loader.translation_status = "ready_mock"
        loader.tts_status = "ready_mock"

        # Transcribe test
        res = loader.transcribe(torch.zeros(16000), language="hi")
        self.assertIn("text", res)
        self.assertEqual(res["text"], "नमस्ते आप कैसे हैं")

        # Translate test
        tr_res = loader.translate("hello", source_lang="eng_Latn", target_lang="hin_Deva")
        self.assertEqual(tr_res, "नमस्ते")

        # Synthesize test
        synth_bytes = loader.synthesize("नमस्ते")
        self.assertIsInstance(synth_bytes, bytes)
        self.assertGreater(len(synth_bytes), 0)

        # Status check
        status = loader.get_status()
        self.assertEqual(status["asr_status"], "ready_mock")


if __name__ == "__main__":
    unittest.main()
