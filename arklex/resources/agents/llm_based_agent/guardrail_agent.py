from collections.abc import Callable

from agents import (
    Agent,
    GuardrailFunctionOutput,
    RunContextWrapper,
    Runner,
    TResponseInputItem,
    input_guardrail,
    output_guardrail,
)
from pydantic import BaseModel

from arklex.utils.logging.logging_utils import LogContext

log_context = LogContext(__name__)


class GuardrailAgentData(BaseModel):
    name: str
    instructions: str


class GuardrailAgentResult(BaseModel):
    instruction_violated: bool
    reasoning: str


def create_input_guardrail_function(name: str, guardrail_agent: Agent) -> Callable:
    """
    Factory function to create a configured input guardrail function.

    Args:
        name: The name to use in the @input_guardrail decorator
        guardrail_agent: The Agent instance to use for guardrail checks

    Returns:
        A decorated async function that performs the guardrail check
    """

    @input_guardrail(name=name)
    async def input_guardrail_func(
        ctx: RunContextWrapper[None],
        agent: Agent,
        input: str | list[TResponseInputItem],
    ) -> GuardrailFunctionOutput:
        result = await Runner.run(guardrail_agent, input, context=ctx.context)
        log_context.info(f"Input guardrail result: {result.final_output}")
        return GuardrailFunctionOutput(
            output_info=result.final_output,
            tripwire_triggered=result.final_output.instruction_violated,
        )

    return input_guardrail_func


def create_output_guardrail_function(name: str, guardrail_agent: Agent) -> Callable:
    """
    Factory function to create a configured output guardrail function.

    Args:
        name: The name to use in the @output_guardrail decorator
        guardrail_agent: The Agent instance to use for guardrail checks

    Returns:
        A decorated async function that performs the guardrail check
    """

    @output_guardrail(name=name)
    async def output_guardrail_func(
        ctx: RunContextWrapper[None], agent: Agent, output: str
    ) -> GuardrailFunctionOutput:
        result = await Runner.run(guardrail_agent, output, context=ctx.context)
        log_context.info(f"Output guardrail result: {result.final_output}")
        return GuardrailFunctionOutput(
            output_info=result.final_output,
            tripwire_triggered=result.final_output.instruction_violated,
        )

    return output_guardrail_func
