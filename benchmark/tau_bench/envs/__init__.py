from benchmark.tau_bench.envs.base import Env
from benchmark.tau_bench.envs.user import UserStrategy


def get_env(
    env_name: str,
    user_strategy: str | UserStrategy,
    user_model: str,
    task_split: str,
    user_provider: str | None = None,
    task_index: int | None = None,
) -> Env:
    if env_name == "retail":
        from benchmark.tau_bench.envs.retail import MockRetailDomainEnv

        return MockRetailDomainEnv(
            user_strategy=user_strategy,
            user_model=user_model,
            task_split=task_split,
            user_provider=user_provider,
            task_index=task_index,
        )
    elif env_name == "airline":
        from benchmark.tau_bench.envs.airline import MockAirlineDomainEnv

        return MockAirlineDomainEnv(
            user_strategy=user_strategy,
            user_model=user_model,
            task_split=task_split,
            user_provider=user_provider,
            task_index=task_index,
        )
    else:
        raise ValueError(f"Unknown environment: {env_name}")
