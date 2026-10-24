
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "app": "tandem",
        "version": "0.1.0",
    }


def test_chat_returns_model_reply():
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
    assert response.json() == {
        "reply": "Tandem is a personal assistant.",
        "model": "qwen2.5:3b",
    }
    mock_generate.assert_awaited_once_with("What is Tandem?")


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
