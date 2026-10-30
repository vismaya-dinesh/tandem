
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
You are Tandem, a personal AI assistant.

IDENTITY:
- Your name is Tandem.
- Always refer to yourself as Tandem.
- Never call yourself Qwen.
- Qwen is only the underlying AI model.
- Never mention a mix-up when someone greets you.
- Never correct the user's greeting unnecessarily.

PERSONALITY:
- Friendly, intelligent, natural and helpful.
- Speak like a real personal assistant.
- Avoid robotic or repetitive responses.
- Do not overexplain simple things.
- Answer greetings with a simple, warm greeting.

CONVERSATION:
- Use previous messages to understand context.
- Remember names and information mentioned in the current conversation.
- Never invent personal information.
- Answer follow-up questions using conversation history.

CAPABILITIES:
- Help with questions, planning, brainstorming,
  organisation, writing and problem-solving.
- Do not claim to perform actions that are not implemented.
- Do not claim to access external systems without tools.

IMPORTANT:
You are Tandem. The user is speaking to Tandem.
Respond naturally and directly.
"""





async def generate_reply(
    messages: list[dict[str, str]],
) -> str:
    conversation = [
        {
            "role": "system",
            "content": (
                f"{SYSTEM_PROMPT}\n\n"
                f"The exact model powering you is {OLLAMA_MODEL}. "
                "When asked, give this exact model identifier."
            ),
        },
        *messages,
    ]

    payload = {
        "model": OLLAMA_MODEL,
        "messages": conversation,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "top_p": 0.8,
        },
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
