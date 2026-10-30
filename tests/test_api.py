
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from backend.main import app
from backend.database import get_conversation


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "app": "tandem",
        "version": "0.2.0",
    }


def test_chat_returns_model_reply_and_conversation_id():
    with patch(
        "backend.main.generate_reply",
        new_callable=AsyncMock,
    ) as mock_generate:
        mock_generate.return_value = "Tandem is a personal assistant."

        response = client.post(
            "/chat",
            json={"message": "What is Tandem?"},
        )

    assert response.status_code == 200

    data = response.json()
    assert data["reply"] == "Tandem is a personal assistant."
    assert data["model"] == "qwen2.5:3b"
    assert data["conversation_id"]

    mock_generate.assert_awaited_once_with(
    [{"role": "user", "content": "What is Tandem?"}]
    )


def test_chat_rejects_empty_message():
    response = client.post(
        "/chat",
        json={"message": ""},
    )

    assert response.status_code == 422


def test_chat_rejects_message_over_4000_characters():
    response = client.post(
        "/chat",
        json={"message": "a" * 4001},
    )

    assert response.status_code == 422


def test_chat_saves_user_and_assistant_messages():
    with patch(
        "backend.main.generate_reply",
        new_callable=AsyncMock,
    ) as mock_generate:
        mock_generate.return_value = "Hello from Tandem!"

        response = client.post(
            "/chat",
            json={"message": "Hello!"},
        )

    assert response.status_code == 200
    conversation_id = response.json()["conversation_id"]

    conversation = get_conversation(conversation_id)

    assert conversation is not None
    assert len(conversation["messages"]) == 2

    assert conversation["messages"][0]["role"] == "user"
    assert conversation["messages"][0]["content"] == "Hello!"

    assert conversation["messages"][1]["role"] == "assistant"
    assert conversation["messages"][1]["content"] == "Hello from Tandem!"


def test_conversation_history_endpoint():
    with patch(
        "backend.main.generate_reply",
        new_callable=AsyncMock,
    ) as mock_generate:
        mock_generate.return_value = "History test reply."

        chat_response = client.post(
            "/chat",
            json={"message": "Remember this conversation."},
        )

    conversation_id = chat_response.json()["conversation_id"]

    history_response = client.get(
        f"/conversations/{conversation_id}"
    )

    assert history_response.status_code == 200

    history = history_response.json()
    assert history["id"] == conversation_id
    assert len(history["messages"]) == 2
    assert history["messages"][0]["content"] == (
        "Remember this conversation."
    )
    assert history["messages"][1]["content"] == (
        "History test reply."
    )


def test_unknown_conversation_returns_404():
    response = client.post(
        "/chat",
        json={
            "message": "Hello",
            "conversation_id": "nonexistent-conversation",
        },
    )

    assert response.status_code == 404


def test_existing_conversation_can_receive_another_message():
    with patch(
        "backend.main.generate_reply",
        new_callable=AsyncMock,
    ) as mock_generate:
        mock_generate.side_effect = [
            "First reply.",
            "Second reply.",
        ]

        first_response = client.post(
            "/chat",
            json={"message": "First message"},
        )

        conversation_id = first_response.json()["conversation_id"]

        second_response = client.post(
            "/chat",
            json={
                "message": "Second message",
                "conversation_id": conversation_id,
            },
        )

    assert second_response.status_code == 200
    assert second_response.json()["conversation_id"] == conversation_id

    conversation = get_conversation(conversation_id)

    assert conversation is not None
    assert len(conversation["messages"]) == 4


def test_chat_passes_previous_messages_as_context():
    with patch(
        "backend.main.generate_reply",
        new_callable=AsyncMock,
    ) as mock_generate:
        mock_generate.side_effect = [
            "Python is a great programming language.",
            "You mentioned Python.",
        ]

        first_response = client.post(
            "/chat",
            json={"message": "My favourite language is Python."},
        )

        conversation_id = first_response.json()["conversation_id"]

        second_response = client.post(
            "/chat",
            json={
                "message": "Which language did I mention?",
                "conversation_id": conversation_id,
            },
        )

    assert second_response.status_code == 200

    second_call_messages = mock_generate.await_args_list[1].args[0]

    assert second_call_messages == [
        {
            "role": "user",
            "content": "My favourite language is Python.",
        },
        {
            "role": "assistant",
            "content": "Python is a great programming language.",
        },
        {
            "role": "user",
            "content": "Which language did I mention?",
        },
    ]


def test_list_conversations_returns_created_conversation():
    with patch(
        "backend.main.generate_reply",
        new_callable=AsyncMock,
    ) as mock_generate:
        mock_generate.return_value = "Hello!"

        chat_response = client.post(
            "/chat",
            json={"message": "Test conversation listing"},
        )

    conversation_id = chat_response.json()["conversation_id"]

    response = client.get("/conversations")

    assert response.status_code == 200

    conversations = response.json()
    matching = [
        item for item in conversations
        if item["id"] == conversation_id
    ]

    assert len(matching) == 1
    assert matching[0]["title"] == "Test conversation listing"
    assert matching[0]["message_count"] == 2


def test_delete_conversation_removes_messages():
    with patch(
        "backend.main.generate_reply",
        new_callable=AsyncMock,
    ) as mock_generate:
        mock_generate.return_value = "Hello!"

        chat_response = client.post(
            "/chat",
            json={"message": "Conversation to delete"},
        )

    conversation_id = chat_response.json()["conversation_id"]

    delete_response = client.delete(
        f"/conversations/{conversation_id}"
    )

    assert delete_response.status_code == 200
    assert delete_response.json()["status"] == "deleted"

    assert get_conversation(conversation_id) is None

    history_response = client.get(
        f"/conversations/{conversation_id}"
    )

    assert history_response.status_code == 404


def test_deleting_one_conversation_preserves_another():
    with patch(
        "backend.main.generate_reply",
        new_callable=AsyncMock,
    ) as mock_generate:
        mock_generate.side_effect = [
            "First reply",
            "Second reply",
        ]

        first_response = client.post(
            "/chat",
            json={"message": "First conversation"},
        )

        second_response = client.post(
            "/chat",
            json={"message": "Second conversation"},
        )

    first_id = first_response.json()["conversation_id"]
    second_id = second_response.json()["conversation_id"]

    delete_response = client.delete(
        f"/conversations/{first_id}"
    )

    assert delete_response.status_code == 200
    assert get_conversation(first_id) is None

    remaining = get_conversation(second_id)

    assert remaining is not None
    assert len(remaining["messages"]) == 2
    assert remaining["messages"][0]["content"] == (
        "Second conversation"
    )
