import argparse
import asyncio
import json

from patterns.registry import dispatch_pattern


def load_config(path: str = "config.json") -> dict:
    with open(path) as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, default="config.json", help="config file path"
    )
    args = parser.parse_args()

    config = load_config(args.config_path)
    asyncio.run(dispatch_pattern(config))


if __name__ == "__main__":
    main()
