import pytest
from unittest.mock import patch
from arklex.orchestrator.NLU.services.api_service import APIClientService
from arklex.utils.exceptions import APIError


@pytest.fixture
def dummy_base_url() -> str:
    return "http://dummy-api"


@pytest.fixture
def api_service(dummy_base_url):
    return APIClientService(base_url=dummy_base_url)


def test_api_client_service_initialization(dummy_base_url) -> None:
    service = APIClientService(base_url=dummy_base_url)
    assert service.base_url == dummy_base_url


def test_predict_intent_success(api_service) -> None:
    with patch.object(
        api_service,
        "_make_request",
        return_value={"intent": "0", "idx2intents_mapping": {"0": "greeting"}},
    ):
        result = api_service.predict_intent(
            text="hello",
            intents={
                "greeting": [
                    {
                        "attribute": {
                            "definition": "A greeting",
                            "sample_utterances": ["hello", "hi"],
                        }
                    }
                ]
            },
            chat_history_str="User: hello",
            model_config={"temperature": 0.7},
        )
        assert result == "greeting"


def test_predict_intent_api_error(api_service) -> None:
    with patch.object(api_service, "_make_request", side_effect=APIError("API error")):
        with pytest.raises(APIError, match="API error"):
            api_service.predict_intent(
                text="hello",
                intents={
                    "greeting": [
                        {
                            "attribute": {
                                "definition": "A greeting",
                                "sample_utterances": ["hello", "hi"],
                            }
                        }
                    ]
                },
                chat_history_str="User: hello",
                model_config={"temperature": 0.7},
            )


def test_predict_intent_missing_intent_returns_others(api_service) -> None:
    with patch.object(api_service, "_make_request", return_value={}):
        result = api_service.predict_intent(
            text="hello",
            intents={
                "greeting": [
                    {
                        "attribute": {
                            "definition": "A greeting",
                            "sample_utterances": ["hello", "hi"],
                        }
                    }
                ]
            },
            chat_history_str="User: hello",
            model_config={"temperature": 0.7},
        )
        assert result == "others"
