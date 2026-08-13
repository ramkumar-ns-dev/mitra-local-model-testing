import unittest
from backend.services.itn_service import ITNService, itn_service, ITN_LANG_MAP


class TestITNService(unittest.TestCase):
    def test_itn_service_initialization(self):
        """Verify ITNService loads and detects supported languages from indic-itn."""
        self.assertTrue(itn_service.available)
        self.assertIsInstance(itn_service.supported_languages, list)
        self.assertGreaterEqual(len(itn_service.supported_languages), 4)
        for lang in ["hi", "kn", "ta", "te"]:
            self.assertIn(lang, itn_service.supported_languages)

    def test_resolve_lang_code(self):
        """Verify language code resolution for various ISO, NLLB, and full name formats."""
        # Hindi variations
        self.assertEqual(itn_service.resolve_lang_code("hi"), "hi")
        self.assertEqual(itn_service.resolve_lang_code("hin"), "hi")
        self.assertEqual(itn_service.resolve_lang_code("hin_Deva"), "hi")
        self.assertEqual(itn_service.resolve_lang_code("hindi"), "hi")

        # Telugu variations
        self.assertEqual(itn_service.resolve_lang_code("te"), "te")
        self.assertEqual(itn_service.resolve_lang_code("tel"), "te")
        self.assertEqual(itn_service.resolve_lang_code("tel_Telu"), "te")
        self.assertEqual(itn_service.resolve_lang_code("telugu"), "te")

        # Kannada variations
        self.assertEqual(itn_service.resolve_lang_code("kn"), "kn")
        self.assertEqual(itn_service.resolve_lang_code("kan"), "kn")
        self.assertEqual(itn_service.resolve_lang_code("kan_Knda"), "kn")
        self.assertEqual(itn_service.resolve_lang_code("kannada"), "kn")

        # Tamil variations
        self.assertEqual(itn_service.resolve_lang_code("ta"), "ta")
        self.assertEqual(itn_service.resolve_lang_code("tam"), "ta")
        self.assertEqual(itn_service.resolve_lang_code("tam_Taml"), "ta")
        self.assertEqual(itn_service.resolve_lang_code("tamil"), "ta")

        # Unsupported or empty
        self.assertIsNone(itn_service.resolve_lang_code("en"))
        self.assertIsNone(itn_service.resolve_lang_code("eng_Latn"))
        self.assertIsNone(itn_service.resolve_lang_code(None))
        self.assertIsNone(itn_service.resolve_lang_code(""))

    def test_hindi_normalization(self):
        """Test Hindi Inverse Text Normalization."""
        raw = "पाँच लाख साठ हजार रुपये"
        normalized = itn_service.normalize(raw, lang="hin_Deva")
        self.assertTrue("₹560000" in normalized or "560000" in normalized)

        time_raw = "पाँच बजकर तीस मिनट"
        normalized_time = itn_service.normalize(time_raw, lang="hi")
        self.assertTrue("5:30" in normalized_time)

    def test_telugu_normalization(self):
        """Test Telugu Inverse Text Normalization."""
        raw = "రెండు వందల రూపాయలు"
        normalized = itn_service.normalize(raw, lang="tel_Telu")
        self.assertTrue("₹200" in normalized or "200" in normalized)

    def test_kannada_normalization(self):
        """Test Kannada Inverse Text Normalization."""
        raw = "ಮೂರು ಬಿಂದು ಒಂದು ನಾಲ್ಕು"
        normalized = itn_service.normalize(raw, lang="kan_Knda")
        self.assertTrue("3.14" in normalized)

        pct_raw = "ಇಪ್ಪತ್ತೈದು ಶೇಕಡಾ"
        normalized_pct = itn_service.normalize(pct_raw, lang="kn")
        self.assertTrue("25%" in normalized_pct)

    def test_tamil_normalization(self):
        """Test Tamil Inverse Text Normalization safely."""
        raw = "வணக்கம்"
        normalized = itn_service.normalize(raw, lang="tam_Taml")
        self.assertIsInstance(normalized, str)

    def test_unsupported_language_fallback(self):
        """Test that unsupported languages fall back cleanly to original text."""
        text = "Hello world 123"
        self.assertEqual(itn_service.normalize(text, lang="en"), text)
        self.assertEqual(itn_service.normalize(text, lang="invalid_lang"), text)

    def test_empty_input_fallback(self):
        """Test handling of empty text input."""
        self.assertEqual(itn_service.normalize("", lang="hi"), "")
        self.assertIsNone(itn_service.normalize(None, lang="hi"))

    def test_itn_status(self):
        """Test get_status method."""
        status = itn_service.get_status()
        self.assertTrue(status["available"])
        self.assertIsInstance(status["supported_languages"], list)
        self.assertGreaterEqual(len(status["supported_languages"]), 4)


if __name__ == "__main__":
    unittest.main()
