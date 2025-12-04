# app/groq_llm.py

import requests
import time
from app.config import get_groq_key

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Use Groq's 2025 stable model
GROQ_DEFAULT_MODEL = "llama-3.1-8b-instant"

def groq_call(prompt: str, model=GROQ_DEFAULT_MODEL):
    api_key = get_groq_key()

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    for attempt in range(3):
        try:
            response = requests.post(GROQ_URL, json=payload, headers=headers, timeout=30)

            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]

            print(f"[Groq] Error {response.status_code}: {response.text}")

        except Exception as e:
            print(f"[Groq] Exception: {str(e)}")

        print(f"[Groq] Retrying ({attempt + 1}/3)...")
        time.sleep(1)

    raise RuntimeError("Groq API failed after 3 attempts.")
