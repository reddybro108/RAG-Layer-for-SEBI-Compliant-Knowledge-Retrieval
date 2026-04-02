# app/config.py

import os
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv()


def get_hf_key() -> str:
    key = os.getenv("HF_API_KEY")
    if not key:
        raise ValueError("HF_API_KEY not found. Add it to .env file.")
    return key


def get_hf_model() -> str:
    return os.getenv("HF_MODEL", "google/flan-t5-large")


def get_embed_model() -> str:
    return os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def disable_broken_local_proxy() -> None:
    """
    Remove known-bad localhost proxy placeholders that break HF/network calls.
    """
    proxy_vars = [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ]
    for key in proxy_vars:
        value = os.getenv(key, "")
        if "127.0.0.1:9" in value:
            os.environ.pop(key, None)
