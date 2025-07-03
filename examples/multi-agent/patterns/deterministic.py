from agents import Runner
from common.agent_loader import build_agents
from rich import print


async def run_deterministic_flow(config: dict) -> None:
    agents = build_agents(config["agents"], config["llm_config"])
    print(f"[bold green]Welcome to {config['role']}! Type 'quit' to exit.[/bold green]")

    while True:
        user_input = input("You: ")
        if user_input.lower() in {"quit", "exit"}:
            print("\n[red]Exiting. Goodbye![/red]")
            return

        input_items = [{"role": "user", "content": user_input}]

        for _idx, agent in enumerate(agents):
            result = await Runner.run(agent, input_items)
            output = result.final_output
            input_items = result.to_input_list()

        print(f"[bold magenta]Bot:[/bold magenta] {output}")
