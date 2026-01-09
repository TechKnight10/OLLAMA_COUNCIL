"""Configuration for the LLM Council using Ollama."""

import os
from dotenv import load_dotenv

load_dotenv()

# No API key needed for Ollama
OPENROUTER_API_KEY = None

# Council members - list of Ollama model names
COUNCIL_MODELS = [
    "deepseek-r1:7b",
    "ministral-3:3b",
    "gemma3:4b",
    "llama3.2:3b",
]

# Chairman model - synthesizes final response
CHAIRMAN_MODEL = "deepseek-r1:7b"

# Ollama API endpoint
OPENROUTER_API_URL = "http://localhost:11434/api/chat"

# Data directory for conversation storage
DATA_DIR = "data/conversations"
