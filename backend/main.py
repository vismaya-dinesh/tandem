
from fastapi import FastAPI, HTTPException
from httpx import HTTPError

from backend.ollama_client import OLLAMA_MODEL, generate_reply
from backend.schemas import ChatRequest, ChatResponse


app = FastAPI(
    title="Tandem API",
    description="A local-first, auditable personal assistant.",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "app": "tandem",
        "version": "0.1.0",
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        reply = await generate_reply(request.message)

        return ChatResponse(
            reply=reply,
            model=OLLAMA_MODEL,
        )

    except HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "Could not communicate with the "
                "local Ollama server."
            ),
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc
