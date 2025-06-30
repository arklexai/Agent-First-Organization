"""Script for analyzing TAU benchmark results.

This script provides functionality to analyze the results of TAU benchmark runs,
including error identification and result processing. It sets up the necessary
environment and runs the error identification process with default parameters
for the retail environment and OpenAI platform.
"""

import argparse
import os
import sys

from dotenv import load_dotenv

from benchmark.tau_bench.auto_error_identification import (
    get_args,
    run_error_identification,
)

root_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(root_dir)
load_dotenv()

if __name__ == "__main__":
    """Main entry point for the TAU benchmark result analysis.
    
    This script runs the error identification process with the following default parameters:
    - Environment: retail
    - Platform: OpenAI
    - Max concurrency: 16
    
    Additional parameters can be provided via command line arguments:
    --results-path: Path to the results directory
    --output-dir: Path to the output directory
    """
    sys.argv += ["--env", "retail"]
    sys.argv += ["--platform", "openai"]
    sys.argv += ["--max-concurrency", "16"]
    args: argparse.Namespace = get_args()
    run_error_identification(args)
