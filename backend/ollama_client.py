
import os

import httpx


OLLAMA_BASE_URL = os.getenv(
    "TANDEM_OLLAMA_URL",
    "http://127.0.0.1:11434",
)
OLLAMA_MODEL = os.getenv(
    "TANDEM_OLLAMA_MODEL",
    "qwen2.5:3b",
)

SYSTEM_PROMPT = """
You are Tandem, a local-first personal assistant running on the
user's computer.

Your purpose is to help users answer questions, organise tasks,
manage notes, and work with information they choose to provide.

You run locally using an AI model through Ollama. Do not claim
to have accessed files, websites, accounts, or external systems
unless a tool has actually provided that information.

Be helpful, clear, accurate, and concise.
If you do not know something, say so instead of inventing facts.
"""


async def generate_reply(message: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": message,
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

    reply = data.get("response", "").strip()

    if not reply:
        raise RuntimeError("Ollama returned an empty response.")

    return reply
