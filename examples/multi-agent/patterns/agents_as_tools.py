from agents import Agent, Runner, trace
from common.agent_loader import build_tool_wrapped_agents
from rich import print


async def run_agents_as_tools_flow(config: dict) -> None:
    print(f"[bold green]Welcome to {config['role']}! Type 'quit' to exit.[/bold green]")

    # Step 1: Build agents and wrap them as tools
    tool_agents, tool_wrappers = build_tool_wrapped_agents(
        config["agents"], config["llm_config"]
    )

    # Step 2: Create orchestrator
    orchestrator_agent = Agent(
        name="OrchestratorAgent",
        instructions=f"You are the orchestrator. Use the tools to complete this task: {config['task']}. Do NOT answer on your own.",
        tools=tool_wrappers,
        model=config["llm_config"]["model"],
    )

    # Step 3: Create synthesizer
    synthesizer_agent = Agent(
        name="SynthesizerAgent",
        instructions="You are a summarizer. Combine the outputs of the tools into a coherent and useful response for the user.",
        model=config["llm_config"]["model"],
    )
    while True:
        user_input = input("You: ")
        if user_input.lower() in {"quit", "exit"}:
            print("\n[red]Exiting. Goodbye![/red]")
            return

        # Run orchestrator to call tools
        with trace(f"{config['role']}"):
            orchestrator_result = await Runner.run(orchestrator_agent, user_input)

            # Run synthesizer to combine final output
            synth_result = await Runner.run(
                synthesizer_agent, orchestrator_result.to_input_list()
            )

        print(f"\n[bold magenta]Bot:[/bold magenta] {synth_result.final_output}")
