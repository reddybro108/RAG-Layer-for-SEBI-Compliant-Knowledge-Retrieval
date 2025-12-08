# app/config.py

import os
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv()

def get_groq_key() -> str:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("GROQ_API_KEY not found. Add it to .env file.")
    return key
