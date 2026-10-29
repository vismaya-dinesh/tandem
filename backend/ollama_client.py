
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

Help users answer questions, organise tasks, and work with
information they provide.

Use the conversation history to understand follow-up questions.
Treat messages with the user role as user input and messages
with the assistant role as your previous replies.

Do not claim to access files, websites, accounts, or external
systems unless a tool has actually provided that information.
Do not claim that features such as reminders or long-term memory
are available unless they have been implemented.

Be helpful, accurate, and concise. If you do not know something,
say so rather than inventing facts.
"""


async def generate_reply(
    messages: list[dict[str, str]],
) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "system": SYSTEM_PROMPT,
        "messages": messages,
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

    reply = data.get("message", {}).get("content", "").strip()

    if not reply:
        raise RuntimeError("Ollama returned an empty response.")

    return reply
