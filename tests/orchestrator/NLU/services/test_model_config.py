def test_configure_response_format_invalid() -> None:
    from arklex.orchestrator.NLU.services.model_config import ModelConfig

    class DummyModel:
        def bind(self, response_format: object) -> "DummyModel":
            return self

    model = DummyModel()
    config = {"llm_provider": "openai"}
    import pytest

    with pytest.raises(ValueError):
        ModelConfig.configure_response_format(model, config, response_format="invalid")
