"""Tests for OpenAI-compatible model catalog loading."""

import json
from unittest.mock import patch

from freecad_ai.llm.client import LLMClient


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self._payload).encode("utf-8")


def _catalog_payload():
    return {
        "object": "list",
        "data": [
            {
                "id": "deepseek-v4-flash",
                "object": "model",
                "owned_by": "deepseek",
            },
            {
                "id": "ollama-cloud-glm-5-3",
                "object": "model",
                "owned_by": "ollama",
            },
            {
                "id": "commandcode-gpt-5-6-terra",
                "object": "model",
                "owned_by": "commandcode",
            },
            {"object": "model"},
        ],
    }


def _health_payload():
    return {
        "ok": True,
        "service": "codex-router-api-forwarder",
        "providers": {
            "deepseek": {"credential_present": True},
            "ollama-cloud": {"credential_present": True},
            "commandcode": {"credential_present": False},
        },
    }


def _responses(request):
    if request.full_url.endswith("/health"):
        return _FakeResponse(_health_payload())
    assert request.full_url.endswith("/models")
    return _FakeResponse(_catalog_payload())


def test_list_models_can_return_the_full_catalog():
    client = LLMClient(
        provider_name="codex-router",
        base_url="http://127.0.0.1:4203/v1",
        api_key="test-key",
        model="",
    )

    with patch(
        "freecad_ai.llm.client.urllib.request.urlopen",
        side_effect=_responses,
    ) as urlopen:
        assert client.list_models(available_only=False) == [
            "deepseek-v4-flash",
            "ollama-cloud-glm-5-3",
            "commandcode-gpt-5-6-terra",
        ]

    requests = [call.args[0] for call in urlopen.call_args_list]
    assert requests[0].full_url == "http://127.0.0.1:4203/v1/models"
    assert requests[1].full_url == "http://127.0.0.1:4203/health"
    assert requests[0].get_header("Authorization") == "Bearer test-key"


def test_list_models_filters_disabled_router_providers():
    client = LLMClient(
        provider_name="codex-router",
        base_url="http://127.0.0.1:4203/v1",
        api_key="test-key",
        model="",
    )

    with patch(
        "freecad_ai.llm.client.urllib.request.urlopen",
        side_effect=_responses,
    ):
        assert client.list_models() == [
            "deepseek-v4-flash",
            "ollama-cloud-glm-5-3",
        ]
