# generation/groq_llm.py

import time
import requests
from typing import Optional
from app.config import get_groq_key

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"   # Current production baseline
MAX_RETRIES = 3
TIMEOUT = 30   # seconds


def _build_headers(api_key: str) -> dict:
    """Assemble authentication headers for Groq API."""
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _build_payload(prompt: str, model: str) -> dict:
    """Construct a Groq-compliant inference payload."""
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }


def groq_call(prompt: str, model: Optional[str] = None) -> str:
    """
    Execute a managed inference request against Groq's chat-completions API.
    Includes structured retries and telemetry-driven error surfacing.
    """
    api_key = get_groq_key()
    model = model or GROQ_MODEL

    headers = _build_headers(api_key)
    payload = _build_payload(prompt, model)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                GROQ_URL,
                json=payload,
                headers=headers,
                timeout=TIMEOUT,
            )

            # Successful execution path
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]

            # Groq returned an actionable error state
            print(
                f"[Groq] Upstream error (attempt {attempt}/{MAX_RETRIES}) "
                f"→ HTTP {response.status_code}: {response.text}"
            )

        except requests.exceptions.Timeout:
            print(
                f"[Groq] Timeout exception on attempt {attempt}/{MAX_RETRIES}. "
                "Network latency exceeded allowable SLA."
            )

        except requests.exceptions.RequestException as e:
            print(
                f"[Groq] Transport-layer exception on attempt "
                f"{attempt}/{MAX_RETRIES}: {e}"
            )

        # Backoff for retry cadence
        if attempt < MAX_RETRIES:
            time.sleep(1)

    # Escalate systemic failure after exhausting retry pipeline
    raise RuntimeError(
        "Groq API request failed after all retry attempts. "
        "Escalation recommended."
    )
