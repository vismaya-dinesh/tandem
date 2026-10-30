
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from httpx import HTTPError

from backend.database import (
    add_message,
    conversation_exists,
    create_conversation,
    get_conversation,
    initialize_database,
    delete_conversation,
    list_conversations,
)
from backend.ollama_client import OLLAMA_MODEL, generate_reply
from backend.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationResponse,
    ConversationListItem,
)
from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_database()
    yield


app = FastAPI(
    title="Tandem API",
    description="A local-first, auditable personal assistant.",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "app": "tandem",
        "version": "0.2.0",
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    conversation_id = request.conversation_id

    if conversation_id is None:
        conversation_id = create_conversation(
            title=request.message[:60].strip() or "New conversation"
        )
    elif not conversation_exists(conversation_id):
        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        )

    add_message(conversation_id, "user", request.message)

    try:
        conversation = get_conversation(conversation_id)

        if conversation is None:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found.",
            )

        messages = [
            {"role": message["role"], "content": message["content"]}
            for message in conversation["messages"]
        ]

        reply = await generate_reply(messages)
    except HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail="Could not communicate with the local Ollama server.",
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    add_message(conversation_id, "assistant", reply)

    return ChatResponse(
        reply=reply,
        model=OLLAMA_MODEL,
        conversation_id=conversation_id,
    )


@app.get(
    "/conversations/{conversation_id}",
    response_model=ConversationResponse,
)
def read_conversation(conversation_id: str):
    conversation = get_conversation(conversation_id)

    if conversation is None:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        )

    return conversation


@app.get(
    "/conversations",
    response_model=list[ConversationListItem],
)
def read_conversations():
    return list_conversations()


@app.delete("/conversations/{conversation_id}")
def remove_conversation(conversation_id: str):
    deleted = delete_conversation(conversation_id)

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        )

    return {
        "status": "deleted",
        "conversation_id": conversation_id,
    }
