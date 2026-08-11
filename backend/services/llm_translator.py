import os
import requests
import json
import time
from typing import Dict, Any, Optional
from backend.utils.logging_config import get_logger

logger = get_logger()

# Predefined rates in USD per 1 Million Tokens
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    # OpenAI Models
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    # Anthropic Models
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    "claude-3-opus": {"input": 15.00, "output": 75.00},
    # Google Gemini Models
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-3.5-flash": {"input": 1.50, "output": 9.00},
    "gemini-3.1-pro": {"input": 2.00, "output": 12.00},
    # DeepSeek Models
    "deepseek-chat": {"input": 0.55, "output": 2.19},
    "deepseek-reasoner": {"input": 0.55, "output": 2.19},
}

# ISO 639-1 / Bhashini language code mapping to full English name for prompt instructions
LANG_CODE_TO_NAME: Dict[str, str] = {
    "en": "English", "hi": "Hindi", "ta": "Tamil", "te": "Telugu",
    "kn": "Kannada", "ml": "Malayalam", "mr": "Marathi", "gu": "Gujarati",
    "bn": "Bengali", "pa": "Punjabi", "or": "Odia", "as": "Assamese",
    "sa": "Sanskrit", "ur": "Urdu", "ne": "Nepali", "ks": "Kashmiri",
    "kok": "Konkani", "sd": "Sindhi", "doi": "Dogri", "mai": "Maithili",
    "mni": "Manipuri", "sat": "Santali", "brx": "Bodo"
}

def translate_via_llm(
    text: str,
    source_lang: str,
    target_lang: str,
    model: str,
    api_keys: Optional[Dict[str, str]] = None,
    usd_to_inr_rate: float = 84.00
) -> Dict[str, Any]:
    """
    Translates text from source_lang to target_lang using the selected cloud LLM.
    Returns a dictionary containing the translation, latency, and detailed token metrics.
    """
    start_time = time.time()
    
    # Standardize language names for clean prompting
    src_lang_name = LANG_CODE_TO_NAME.get(source_lang, source_lang)
    tgt_lang_name = LANG_CODE_TO_NAME.get(target_lang, target_lang)
    
    # Fetch key inputs
    keys = api_keys or {}
    
    # 1. Determine model category/provider and standard system instructions
    prompt_system = (
        "You are an expert translator. Translate the user's text from "
        f"{src_lang_name} to {tgt_lang_name}. Return ONLY the direct translation text. "
        "Do NOT add any notes, conversational introductions, greetings, quotes, explanations, "
        "or markdown wrapping code blocks. Your output must be purely the translated text."
    )
    
    prompt_user = text
    
    # Find matching model rates
    pricing = MODEL_PRICING.get(model, {"input": 0.0, "output": 0.0})
    input_rate = pricing["input"]
    output_rate = pricing["output"]
    
    translation_text = ""
    input_tokens = 0
    output_tokens = 0
    
    try:
        if model.startswith("gpt-"):
            # OpenAI API call
            api_key = keys.get("openai") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API Key is missing. Please configure it in the Cloud LLM settings.")
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": prompt_system},
                    {"role": "user", "content": prompt_user}
                ],
                "temperature": 0.3
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=20.0
            )
            response.raise_for_status()
            resp_data = response.json()
            
            translation_text = resp_data["choices"][0]["message"]["content"].strip()
            input_tokens = resp_data["usage"]["prompt_tokens"]
            output_tokens = resp_data["usage"]["completion_tokens"]
            
        elif model.startswith("claude-"):
            # Anthropic Claude API call
            api_key = keys.get("anthropic") or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("Anthropic API Key is missing. Please configure it in the Cloud LLM settings.")
            
            # Map simplified model names to actual Anthropic model tags
            model_map = {
                "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
                "claude-3-haiku": "claude-3-haiku-20240307",
                "claude-3-opus": "claude-3-opus-20240229"
            }
            anthropic_model = model_map.get(model, model)
            
            headers = {
                "content-type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            }
            payload = {
                "model": anthropic_model,
                "max_tokens": 1024,
                "system": prompt_system,
                "messages": [
                    {"role": "user", "content": prompt_user}
                ],
                "temperature": 0.3
            }
            
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
                timeout=20.0
            )
            response.raise_for_status()
            resp_data = response.json()
            
            translation_text = resp_data["content"][0]["text"].strip()
            input_tokens = resp_data["usage"]["input_tokens"]
            output_tokens = resp_data["usage"]["output_tokens"]
            
        elif model.startswith("gemini-"):
            # Google Gemini API call
            api_key = keys.get("gemini") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("Gemini API Key is missing. Please configure it in the Cloud LLM settings.")
            
            # Map model names to Google Gemini API format
            headers = {
                "Content-Type": "application/json"
            }
            
            # Incorporate prompt system instructions as a system instruction configuration
            payload = {
                "contents": [{
                    "role": "user",
                    "parts": [{"text": prompt_user}]
                }],
                "systemInstruction": {
                    "parts": [{"text": prompt_system}]
                },
                "generationConfig": {
                    "temperature": 0.3
                }
            }
            
            gemini_model_tag = model
            # Standardize names
            if model == "gemini-1.5-flash":
                gemini_model_tag = "gemini-1.5-flash"
            elif model == "gemini-1.5-pro":
                gemini_model_tag = "gemini-1.5-pro"
            elif model == "gemini-2.5-flash":
                gemini_model_tag = "gemini-2.5-flash"
            elif model == "gemini-3.5-flash":
                gemini_model_tag = "gemini-3.5-flash"
            elif model == "gemini-3.1-pro":
                gemini_model_tag = "gemini-3.1-pro"
            
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model_tag}:generateContent?key={api_key}"
            
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=20.0
            )
            response.raise_for_status()
            resp_data = response.json()
            
            translation_text = resp_data["candidates"][0]["content"]["parts"][0]["text"].strip()
            # Parse usage metadata
            usage = resp_data.get("usageMetadata", {})
            input_tokens = usage.get("promptTokenCount", 0)
            output_tokens = usage.get("candidatesTokenCount", 0)
            
        elif model.startswith("deepseek-"):
            # DeepSeek API call
            api_key = keys.get("deepseek") or os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError("DeepSeek API Key is missing. Please configure it in the Cloud LLM settings.")
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": prompt_system},
                    {"role": "user", "content": prompt_user}
                ],
                "temperature": 0.3
            }
            
            response = requests.post(
                "https://api.deepseek.com/chat/completions",
                headers=headers,
                json=payload,
                timeout=30.0
            )
            response.raise_for_status()
            resp_data = response.json()
            
            translation_text = resp_data["choices"][0]["message"]["content"].strip()
            input_tokens = resp_data["usage"]["prompt_tokens"]
            output_tokens = resp_data["usage"]["completion_tokens"]
            
        else:
            raise ValueError(f"Unsupported Cloud LLM model: {model}")
            
    except requests.exceptions.RequestException as req_err:
        logger.error(f"HTTP request error during {model} API translation: {req_err}")
        detail = str(req_err)
        try:
            if req_err.response is not None:
                err_json = req_err.response.json()
                if "error" in err_json:
                    detail = err_json["error"].get("message") or str(err_json["error"])
        except Exception:
            pass
        raise RuntimeError(f"Cloud LLM Translation failed ({model}): {detail}")
    except Exception as exc:
        logger.error(f"Error during {model} API translation: {exc}")
        raise RuntimeError(str(exc))
        
    latency_ms = int((time.time() - start_time) * 1000)
    
    # Calculate costs
    cost_usd = ((input_tokens * input_rate) / 1_000_000) + ((output_tokens * output_rate) / 1_000_000)
    cost_inr = cost_usd * usd_to_inr_rate
    
    return {
        "translation": translation_text,
        "latency_ms": latency_ms,
        "token_metrics": {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost_usd": cost_usd,
            "cost_inr": cost_inr
        }
    }
