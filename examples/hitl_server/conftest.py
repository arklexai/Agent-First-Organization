import argparse


def pytest_addoption(parser: argparse.ArgumentParser) -> None:
    parser.addoption("--model", action="store", default="gpt-4o-mini", help="LLM model")
    parser.addoption(
        "--llm_provider",
        action="store",
        default="openai",
        help="LLM provider",
    )
