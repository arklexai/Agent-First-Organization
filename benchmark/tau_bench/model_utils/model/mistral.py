import os
from typing import Dict, List, Optional

from benchmark.tau_bench.model_utils.api.datapoint import Datapoint
from benchmark.tau_bench.model_utils.model.chat import ChatModel, Message
from benchmark.tau_bench.model_utils.model.completion import (
    approx_cost_for_datapoint,
    approx_prompt_str,
)
from benchmark.tau_bench.model_utils.model.general_model import wrap_temperature
from benchmark.tau_bench.model_utils.model.utils import approx_num_tokens

DEFAULT_MISTRAL_MODEL = "mistral-large-latest"

PRICE_PER_INPUT_TOKEN_MAP: Dict[str, float] = {
    "mistral-largest-latest": 3 / 1000000,
}
INPUT_PRICE_PER_TOKEN_FALLBACK = 10 / 1000000

CAPABILITY_SCORE_MAP: Dict[str, float] = {
    "mistral-largest-latest": 0.9,
}
CAPABILITY_SCORE_FALLBACK = 0.3

# TODO: implement
LATENCY_MS_PER_OUTPUT_TOKEN_MAP: Dict[str, float] = {}
# TODO: implement
LATENCY_MS_PER_OUTPUT_TOKEN_FALLBACK = 0.0

MAX_CONTEXT_LENGTH_MAP: Dict[str, int] = {
    "mistral-largest-latest": 128000,
}
MAX_CONTEXT_LENGTH_FALLBACK = 128000


class MistralModel(ChatModel):
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
    ) -> None:
        from mistralai.async_client import MistralAsyncClient
        from mistralai.client import MistralClient

        if model is None:
            self.model = DEFAULT_MISTRAL_MODEL
        else:
            self.model = model

        api_key = None
        if api_key is None:
            api_key = os.getenv("MISTRAL_API_KEY")
            if api_key is None:
                raise ValueError("MISTRAL_API_KEY environment variable is not set")
        self.client = MistralClient(api_key=api_key)
        self.async_client = MistralAsyncClient(api_key=api_key)
        self.temperature = temperature

    def generate_message(
        self,
        messages: List[Message],
        force_json: bool,
        temperature: Optional[float] = None,
    ) -> Message:
        if temperature is None:
            temperature = self.temperature
        msgs = self.build_generate_message_state(messages)
        res = self.client.chat(
            model=self.model,
            messages=msgs,
            temperature=wrap_temperature(temperature),
            response_format={"type": "json_object" if force_json else "text"},
        )
        return self.handle_generate_message_response(
            prompt=msgs, content=res.choices[0].message.content, force_json=force_json
        )

    def get_approx_cost(self, dp: Datapoint) -> float:
        cost_per_token = PRICE_PER_INPUT_TOKEN_MAP.get(
            self.model, INPUT_PRICE_PER_TOKEN_FALLBACK
        )
        return approx_cost_for_datapoint(dp=dp, price_per_input_token=cost_per_token)

    def get_latency(self, dp: Datapoint) -> float:
        latency_per_output_token = LATENCY_MS_PER_OUTPUT_TOKEN_MAP.get(
            self.model, LATENCY_MS_PER_OUTPUT_TOKEN_FALLBACK
        )
        return approx_cost_for_datapoint(
            dp=dp, price_per_input_token=latency_per_output_token
        )

    def get_capability(self) -> float:
        return CAPABILITY_SCORE_MAP.get(self.model, CAPABILITY_SCORE_FALLBACK)

    def supports_dp(self, dp: Datapoint) -> bool:
        prompt = approx_prompt_str(dp)
        return approx_num_tokens(prompt) <= MAX_CONTEXT_LENGTH_MAP.get(
            self.model, MAX_CONTEXT_LENGTH_FALLBACK
        )
