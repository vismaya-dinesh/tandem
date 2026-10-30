
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    model: str
    conversation_id: str


class MessageResponse(BaseModel):
    id: int
    role: str
    content: str
    created_at: str


class ConversationResponse(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    messages: list[MessageResponse]


class ConversationListItem(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int
