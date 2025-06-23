from arklex.env.env import Environment
from arklex.orchestrator.NLU.services.model_service import DummyModelService


def test_environment_uses_dummy_model_service() -> None:
    env = Environment(tools=[], workers=[])
    assert isinstance(env.model_service, DummyModelService)
