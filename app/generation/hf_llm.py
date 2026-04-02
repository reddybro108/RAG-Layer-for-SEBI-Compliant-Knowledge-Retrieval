# generation/groq_llm.py

import time
from typing import Optional

import requests

from app.config import get_hf_key, get_hf_model

HF_API_URL = "https://router.huggingface.co/hf-inference/models/{model}"
MAX_RETRIES = 3
TIMEOUT = 30


def _build_headers(api_key: str) -> dict:
    """Assemble authentication headers for Hugging Face router API."""
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _build_payload(prompt: str, model: str) -> dict:
    """Construct a Hugging Face text-generation payload."""
    return {
        "inputs": prompt,
        "parameters": {
            "temperature": 0.2,
            "max_new_tokens": 700,
            "return_full_text": False,
        },
        "options": {
            "wait_for_model": True,
            "use_cache": True,
        },
    }


def _extract_text(data) -> str:
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict) and "generated_text" in first:
            return first["generated_text"]

    if isinstance(data, dict):
        if "generated_text" in data:
            return data["generated_text"]
        if "error" in data:
            raise RuntimeError(f"Hugging Face API error: {data['error']}")

    raise RuntimeError(f"Unexpected Hugging Face response shape: {data}")


def hf_call(prompt: str, model: Optional[str] = None) -> str:
    """Execute a Hugging Face inference request with basic retries."""
    api_key = get_hf_key()
    model = model or get_hf_model()

    headers = _build_headers(api_key)
    payload = _build_payload(prompt, model)
    url = HF_API_URL.format(model=model)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=TIMEOUT,
            )

            if response.status_code == 200:
                data = response.json()
                return _extract_text(data)

            print(
                f"[HuggingFace] Upstream error (attempt {attempt}/{MAX_RETRIES}) "
                f"-> HTTP {response.status_code}: {response.text}"
            )

        except requests.exceptions.Timeout:
            print(
                f"[HuggingFace] Timeout exception on attempt {attempt}/{MAX_RETRIES}. "
                "Network latency exceeded the configured timeout."
            )

        except requests.exceptions.RequestException as exc:
            print(
                f"[HuggingFace] Transport-layer exception on attempt "
                f"{attempt}/{MAX_RETRIES}: {exc}"
            )

        if attempt < MAX_RETRIES:
            time.sleep(1)

    raise RuntimeError(
        "Hugging Face API request failed after all retry attempts. "
        "Escalation recommended."
    )
