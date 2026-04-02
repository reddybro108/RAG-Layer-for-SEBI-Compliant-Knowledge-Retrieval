import re
import time
from typing import Optional

import requests
from huggingface_hub import InferenceClient

from app.config import disable_broken_local_proxy, get_hf_key, get_hf_model

HF_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"
HF_INFERENCE_URL = "https://router.huggingface.co/hf-inference/models/{model}"
MAX_RETRIES = 3
TIMEOUT = 60


def _build_headers(api_key: str) -> dict:
    """Assemble authentication headers for Hugging Face router API."""
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _build_chat_payload(prompt: str, model: str) -> dict:
    """Construct an OpenAI-compatible chat completion payload for HF router."""
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 700,
    }


def _extract_chat_text(data) -> str:
    if isinstance(data, dict):
        choices = data.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            content = message.get("content")
            if isinstance(content, str):
                return content
        if "error" in data:
            raise RuntimeError(f"Hugging Face API error: {data['error']}")

    raise RuntimeError(f"Unexpected Hugging Face chat response shape: {data}")


def _hf_inference_generation(prompt: str, model: str, api_key: str) -> str:
    """Run text generation via Hugging Face SDK for text2text models like FLAN."""
    client = InferenceClient(model=model, token=api_key)
    text = client.text_generation(
        prompt,
        max_new_tokens=700,
        temperature=0.2,
        return_full_text=False,
    )
    if not isinstance(text, str):
        raise RuntimeError(f"Unexpected HF SDK response shape: {text}")
    return text


def _hf_inference_generation_http(prompt: str, model: str, api_key: str) -> str:
    """Run text generation via HF router inference endpoint (SDK fallback)."""
    url = HF_INFERENCE_URL.format(model=model)
    payload = {
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
    headers = _build_headers(api_key)
    response = requests.post(url, json=payload, headers=headers, timeout=TIMEOUT)
    if response.status_code != 200:
        raise RuntimeError(f"HF inference HTTP {response.status_code}: {response.text}")

    data = response.json()
    if isinstance(data, list) and data and isinstance(data[0], dict):
        generated = data[0].get("generated_text")
        if isinstance(generated, str):
            return generated

    if isinstance(data, dict):
        generated = data.get("generated_text")
        if isinstance(generated, str):
            return generated
        if "error" in data:
            raise RuntimeError(f"Hugging Face API error: {data['error']}")

    raise RuntimeError(f"Unexpected HF inference response shape: {data}")


def _local_rag_fallback(prompt: str, reason: str) -> str:
    """Generate a deterministic retrieval-only answer when remote LLM calls fail."""
    pattern = re.compile(
        r"\[(?P<source>[^\]]+?) - chunk (?P<chunk>\d+)\]:\n(?P<text>.*?)(?=\n\n\[|\Z)",
        re.DOTALL,
    )
    matches = list(pattern.finditer(prompt))

    if not matches:
        marker = "Retrieved SEBI Context:"
        if marker in prompt:
            raw = prompt.split(marker, 1)[1].strip()
            snippet = re.sub(r"\s+", " ", raw)[:800].rstrip()
            if len(raw) > 800:
                snippet += "..."
            return (
                "Generation provider unavailable; returning retrieval-only context excerpt.\n"
                f"Failure detail: {reason}\n\n{snippet}"
            )
        return (
            "Generation is temporarily unavailable and no parsed context was found for "
            "fallback synthesis."
        )

    best = matches[0]
    source = best.group("source").strip()
    chunk = best.group("chunk").strip()
    text = re.sub(r"\s+", " ", best.group("text").strip())
    snippet = text[:700].rstrip()
    if len(text) > 700:
        snippet += "..."

    lines = [
        "Generation provider unavailable; returning best-evidence answer.",
        f"Failure detail: {reason}",
        "",
        f"{snippet} [{source} - chunk {chunk}]",
    ]
    return "\n".join(lines)


def hf_call(prompt: str, model: Optional[str] = None) -> str:
    """Execute a Hugging Face request with retries and local fallback."""
    disable_broken_local_proxy()
    model = model or get_hf_model()

    try:
        api_key = get_hf_key()
    except Exception as exc:
        return _local_rag_fallback(prompt, f"Missing HF key: {exc}")

    use_sdk_inference = model.lower().startswith("google/flan-")

    if use_sdk_inference:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return _hf_inference_generation(prompt, model, api_key)
            except Exception as exc:
                print(
                    f"[HuggingFace] SDK error (attempt {attempt}/{MAX_RETRIES}) "
                    f"for model '{model}': {type(exc).__name__}: {exc!r}"
                )
                if attempt < MAX_RETRIES:
                    time.sleep(1)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return _hf_inference_generation_http(prompt, model, api_key)
            except Exception as exc:
                print(
                    f"[HuggingFace] HTTP inference fallback error "
                    f"(attempt {attempt}/{MAX_RETRIES}) for model '{model}': "
                    f"{type(exc).__name__}: {exc!r}"
                )
                if attempt < MAX_RETRIES:
                    time.sleep(1)

        return _local_rag_fallback(
            prompt,
            f"FLAN inference unavailable for model '{model}'",
        )

    headers = _build_headers(api_key)
    payload = _build_chat_payload(prompt, model)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                HF_CHAT_URL,
                json=payload,
                headers=headers,
                timeout=TIMEOUT,
            )

            if response.status_code == 200:
                data = response.json()
                return _extract_chat_text(data)

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

    return _local_rag_fallback(
        prompt,
        f"Chat completion unavailable for model '{model}'",
    )
