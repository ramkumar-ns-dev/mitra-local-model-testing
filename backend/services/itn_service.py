"""
ITN (Inverse Text Normalization) Service for Indic languages using indic-itn (v0.2.1+).
Supports Hindi (hi), Telugu (te), Kannada (kn), Tamil (ta), and other languages supported by indic-itn.
"""

from typing import Any, Dict, List, Optional
from backend.utils.logging_config import get_logger

logger = get_logger()

try:
    from indic_itn import IndicITN, get_supported_languages
    INDIC_ITN_AVAILABLE = True
except ImportError:
    INDIC_ITN_AVAILABLE = False
    IndicITN = None
    get_supported_languages = None

# Comprehensive mapping of common language codes/tags to standard 2-letter ISO codes supported by indic-itn
ITN_LANG_MAP: Dict[str, str] = {
    # Hindi
    "hi": "hi",
    "hin": "hi",
    "hindi": "hi",
    "hin_deva": "hi",
    # Telugu
    "te": "te",
    "tel": "te",
    "telugu": "te",
    "tel_telu": "te",
    # Kannada
    "kn": "kn",
    "kan": "kn",
    "kannada": "kn",
    "kan_knda": "kn",
    # Tamil
    "ta": "ta",
    "tam": "ta",
    "tamil": "ta",
    "tam_taml": "ta",
}


class ITNService:
    """
    Inverse Text Normalization Manager supporting dynamic multi-lingual normalization.
    """

    def __init__(self):
        self.available = INDIC_ITN_AVAILABLE
        self._normalizers: Dict[str, Any] = {}
        self.supported_languages: List[str] = []

        if self.available and get_supported_languages is not None:
            try:
                self.supported_languages = get_supported_languages()
                logger.info(
                    f"indic-itn package initialized. Supported languages: {self.supported_languages}"
                )
            except Exception as err:
                logger.warning(f"Error checking indic-itn supported languages: {err}")
                self.supported_languages = ["hi", "kn", "ta", "te"]
        else:
            logger.warning(
                "indic-itn package is not installed or unavailable. ITN normalization disabled."
            )

    def resolve_lang_code(self, lang_input: Optional[str]) -> Optional[str]:
        """
        Resolves input language code or tag to canonical 2-letter language code if supported.
        """
        if not lang_input:
            return None

        cleaned = str(lang_input).strip().lower()

        # Direct map check
        if cleaned in ITN_LANG_MAP:
            candidate = ITN_LANG_MAP[cleaned]
            if candidate in self.supported_languages:
                return candidate

        # Extract prefix if tag contains underscore (e.g., hin_Deva -> hin)
        if "_" in cleaned:
            prefix = cleaned.split("_")[0]
            if prefix in ITN_LANG_MAP:
                candidate = ITN_LANG_MAP[prefix]
                if candidate in self.supported_languages:
                    return candidate

        # Fallback check if cleaned itself is in supported languages
        if cleaned in self.supported_languages:
            return cleaned

        return None

    def get_normalizer(self, lang_code: str) -> Optional[object]:
        """
        Retrieves or instantiates an IndicITN normalizer for the specified language code.
        """
        if not self.available or not IndicITN:
            return None

        code = self.resolve_lang_code(lang_code)
        if not code:
            return None

        if code not in self._normalizers:
            try:
                logger.info(f"Instantiating IndicITN normalizer for language: '{code}'")
                self._normalizers[code] = IndicITN(lang=code)
            except Exception as err:
                logger.error(f"Failed to instantiate IndicITN for '{code}': {err}")
                return None

        return self._normalizers.get(code)

    def normalize(self, text: str, lang: Optional[str] = None) -> str:
        """
        Applies Inverse Text Normalization to the given text using the specified language.
        If ITN is unavailable, unsupported for the language, or fails, returns original text.
        """
        if not text or not text.strip():
            return text

        if not self.available:
            return text

        code = self.resolve_lang_code(lang)
        if not code:
            return text

        normalizer = self.get_normalizer(code)
        if normalizer is None:
            return text

        try:
            normalized_text = normalizer.normalize(text)
            logger.info(f"Applied ITN ({code}): '{text}' -> '{normalized_text}'")
            return normalized_text
        except Exception as err:
            logger.error(f"Failed to apply ITN ({code}) normalization: {err}")
            return text

    def get_status(self) -> Dict[str, Any]:
        """
        Returns status information about ITN service.
        """
        return {
            "available": self.available,
            "supported_languages": self.supported_languages,
            "active_normalizers": list(self._normalizers.keys()),
        }


# Global instance for app-wide use
itn_service = ITNService()
