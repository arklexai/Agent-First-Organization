import os
import sys
import argparse

from dotenv import load_dotenv

from benchmark.tau_bench.auto_error_identification import (
    get_args,
    run_error_identification,
)

root_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(root_dir)
load_dotenv()

if __name__ == "__main__":
    """
    Provide --results-path and --output-dir
    """
    sys.argv += ["--env", "retail"]
    sys.argv += ["--platform", "openai"]
    sys.argv += ["--max-concurrency", "16"]
    args: argparse.Namespace = get_args()
    run_error_identification(args)
