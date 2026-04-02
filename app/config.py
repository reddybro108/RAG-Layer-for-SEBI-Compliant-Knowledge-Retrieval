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
    return os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
