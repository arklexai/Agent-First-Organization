"""Task graph generation and worker initialization for the Arklex framework.

This module provides functionality for generating task graphs and initializing
workers in the Arklex framework. It includes utilities for creating task graphs
based on configuration files, setting up workers like FaissRAGWorker and
DataBaseWorker, and managing the overall system initialization process.
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import time
import zipfile
from typing import Any

from dotenv import load_dotenv

from arklex.env.tools.database.build_database import build_database
from arklex.env.tools.RAG.build_rag import build_rag
from arklex.orchestrator.generator.generator import Generator
from arklex.utils.loader import Loader
from arklex.utils.logging_utils import LogContext
from arklex.utils.model_provider_config import LLM_PROVIDERS
from arklex.utils.provider_utils import (
    get_provider_config,
    validate_and_get_model_class,
)

log_context = LogContext(__name__)
load_dotenv()


def generate_taskgraph(args: argparse.Namespace) -> None:
    """Generate a task graph based on the provided configuration.

    This function initializes a language model, loads the configuration, and uses the Generator
    to create a task graph. It then saves the task graph to a file and updates it with API URLs.

    Args:
        args (argparse.Namespace): Command-line arguments containing configuration and output settings.
    """
    # Validate API key before proceeding
    try:
        provider_config = get_provider_config(args.llm_provider, args.model)
        log_context.info(f"✅ API key for {args.llm_provider} provider is configured")
    except ValueError as e:
        log_context.error(f"❌ API key validation failed: {e}")
        log_context.error(
            "💡 Please ensure your .env file contains the correct API key."
        )
        log_context.error(
            f"   Required environment variable: {args.llm_provider.upper()}_API_KEY"
        )
        return
    except Exception as e:
        log_context.error(f"❌ Unexpected error during API key validation: {e}")
        return

    # Create a temporary config object for validation
    temp_config = type("TempConfig", (), {"llm_provider": args.llm_provider})()
    model_class = validate_and_get_model_class(temp_config)

    # Initialize model with proper API key validation
    if args.llm_provider == "huggingface":
        model = model_class(model=args.model, timeout=30000)
    elif args.llm_provider == "google":
        # Google models use google_api_key parameter
        model = model_class(
            model=args.model, google_api_key=provider_config["api_key"], timeout=30000
        )
    else:
        # Other providers use api_key parameter
        model = model_class(
            model=args.model, api_key=provider_config["api_key"], timeout=30000
        )

    with open(args.config) as f:
        config: dict[str, Any] = json.load(f)
    generator = Generator(config, model, args.output_dir)
    taskgraph = generator.generate()
    taskgraph_filepath: str = generator.save_task_graph(taskgraph)
    # Update the task graph with the API URLs
    with open(os.path.join(os.path.dirname(__file__), taskgraph_filepath)) as f:
        task_graph: dict[str, Any] = json.load(f)
    task_graph["nluapi"] = ""
    task_graph["slotfillapi"] = ""
    with open(taskgraph_filepath, "w") as f:
        json.dump(task_graph, f, indent=4)


def init_worker(args: argparse.Namespace) -> None:
    """Initialize workers based on the provided configuration.

    This function initializes and sets up various workers required by the Arklex framework
    based on the configuration file. It supports different types of workers including:
    - FaissRAGWorker: For RAG (Retrieval-Augmented Generation) functionality
    - DataBaseWorker: For database operations including search, booking, and cancellation

    The function reads the worker configurations from the provided config file and
    initializes the appropriate workers based on their names. Each worker type
    requires specific setup procedures and dependencies.

    Args:
        args (argparse.Namespace): Command-line arguments containing:
            - config: Path to the configuration file
            - output_dir: Directory where worker data will be stored
    """
    # Load configuration from the specified file
    with open(args.config) as f:
        config: dict[str, Any] = json.load(f)
    workers: list[dict[str, Any]] = config["workers"]
    worker_names: set[str] = {worker["name"] for worker in workers}

    # Initialize FaissRAGWorker if specified in configuration
    if "FaissRAGWorker" in worker_names:
        log_context.info("Initializing FaissRAGWorker...")
        build_rag(args.output_dir, config["rag_docs"])

    # Initialize DataBaseWorker and related workers if specified
    elif any(
        node in worker_names
        for node in (
            "DataBaseWorker",
            "search_show",
            "book_show",
            "check_booking",
            "cancel_booking",
        )
    ):
        log_context.info("Initializing DataBaseWorker...")
        build_database(args.output_dir)


def load_documents(
    config: dict[str, Any], document_dir: str | None = None
) -> list[dict[str, str]]:
    """Load documents from various sources specified in the config.

    Args:
        config (Dict[str, Any]): Configuration containing document sources
        document_dir (Optional[str]): Directory containing documents

    Returns:
        List[Dict[str, str]]: List of loaded documents
    """
    loader = Loader()
    all_docs = []
    total_docs_processed = 0
    start_time = time.time()

    # Process all document types consistently
    doc_types = ["instructions", "task_docs", "rag_docs"]
    for doc_type in doc_types:
        if doc_type in config:
            docs = config[doc_type]
            if isinstance(docs, list):
                log_context.info(
                    f"📚 Processing {len(docs)} {doc_type.replace('_', ' ')}..."
                )
                for i, doc in enumerate(docs, 1):
                    source = doc.get("source")
                    doc_type_name = doc.get("type", "text")
                    num_docs = doc.get("num", 1)

                    log_context.info(f"  📄 Document {i}/{len(docs)}: {source}")

                    try:
                        if doc_type_name == "url":
                            log_context.info(
                                f"    🌐 Discovering up to {num_docs} URLs..."
                            )
                            urls = loader.get_all_urls(source, num_docs)
                            log_context.info(
                                f"    📥 Crawling {len(urls)} discovered URLs..."
                            )
                            crawled_docs = loader.to_crawled_url_objs(urls)
                            successful_docs = [
                                doc for doc in crawled_docs if not doc.is_error
                            ]
                            all_docs.extend(crawled_docs)
                            total_docs_processed += len(crawled_docs)
                            log_context.info(
                                f"    ✅ Successfully processed {len(successful_docs)}/{len(crawled_docs)} URLs"
                            )
                        elif doc_type_name == "file":
                            if os.path.isfile(source):
                                if source.lower().endswith(".zip"):
                                    log_context.info("    📦 Extracting ZIP archive...")
                                    with tempfile.TemporaryDirectory() as temp_dir:
                                        with zipfile.ZipFile(source, "r") as zip_ref:
                                            zip_ref.extractall(temp_dir)
                                        file_list = []
                                        for root, _, files in os.walk(temp_dir):
                                            for file in files:
                                                file_list.append(
                                                    os.path.join(root, file)
                                                )
                                        all_docs.extend(
                                            loader.to_crawled_local_objs(file_list)
                                        )
                                        total_docs_processed += len(file_list)
                                        log_context.info(
                                            f"    ✅ Extracted and processed {len(file_list)} files"
                                        )
                                else:
                                    log_context.info("    📄 Processing single file...")
                                    all_docs.extend(
                                        loader.to_crawled_local_objs([source])
                                    )
                                    total_docs_processed += 1
                                    log_context.info(
                                        "    ✅ File processed successfully"
                                    )
                            elif os.path.isdir(source):
                                log_context.info(
                                    "    📁 Processing directory contents..."
                                )
                                file_list = [
                                    os.path.join(source, f) for f in os.listdir(source)
                                ]
                                all_docs.extend(loader.to_crawled_local_objs(file_list))
                                total_docs_processed += len(file_list)
                                log_context.info(
                                    f"    ✅ Processed {len(file_list)} files from directory"
                                )
                            else:
                                raise FileNotFoundError(
                                    f"Source path '{source}' does not exist"
                                )
                        elif doc_type_name == "text":
                            log_context.info("    📝 Processing text content...")
                            all_docs.extend(loader.to_crawled_text([source]))
                            total_docs_processed += 1
                            log_context.info("    ✅ Text content processed")
                        else:
                            raise ValueError(
                                f"Unsupported document type: {doc_type_name}"
                            )
                    except Exception as e:
                        log_context.error(
                            f"❌ Error processing document {source}: {str(e)}"
                        )
                        continue

    elapsed_time = time.time() - start_time
    log_context.info(
        f"📊 Document processing complete: {total_docs_processed} documents in {elapsed_time:.1f}s"
    )

    # Convert CrawledObjects to dictionaries
    return [doc.to_dict() for doc in all_docs]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a task graph from a config file"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for cache and results",
    )
    parser.add_argument(
        "--allow-nested-graph",
        action="store_true",
        default=True,
        help="Whether to allow nested graph generation (default: True)",
    )
    parser.add_argument(
        "--no-nested-graph",
        action="store_true",
        help="Disable nested graph generation (equivalent to --allow-nested-graph=False)",
    )
    parser.add_argument(
        "--no-ui",
        action="store_true",
        help="Disable interactive task editor (task editor is enabled by default)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model to use for generation (e.g., gpt-4o, claude-3-5-haiku-20241022, gemini-1.5-flash)",
    )
    parser.add_argument(
        "--llm_provider",
        type=str,
        default="openai",
        choices=LLM_PROVIDERS,
        help="LLM provider to use (openai, anthropic, google, huggingface)",
    )
    args = parser.parse_args()

    # Handle nested graph flag
    allow_nested_graph = args.allow_nested_graph and not args.no_nested_graph

    # Handle UI flag
    interactable_with_user = not args.no_ui

    # Set up logging
    log_context.setLevel(getattr(logging, args.log_level.upper()))

    # Early API key validation - terminate if API key is not provided
    log_context.info("🔑 Validating API key configuration...")
    try:
        # This will raise ValueError if API key is missing or empty
        provider_config = get_provider_config(args.llm_provider, args.model)
        log_context.info(f"✅ API key for {args.llm_provider} provider is configured")
    except ValueError as e:
        log_context.error(f"❌ API key validation failed: {e}")
        log_context.error(
            "💡 Please ensure your .env file contains the correct API key."
        )
        log_context.error(
            f"   Required environment variable: {args.llm_provider.upper()}_API_KEY"
        )
        sys.exit(1)
    except Exception as e:
        log_context.error(f"❌ Unexpected error during API key validation: {e}")
        sys.exit(1)

    log_context.info("🚀 Starting task graph generation...")
    start_time = time.time()

    # Load config
    log_context.info(f"📋 Loading configuration from {args.config}")
    with open(args.config) as f:
        config = json.load(f)

    # Load documents
    log_context.info("📚 Loading and processing documents...")
    documents = load_documents(config)
    log_context.info(f"📄 Loaded {len(documents)} documents successfully")

    # Instantiate model with proper provider configuration
    log_context.info(
        f"🤖 Initializing language model (provider: {args.llm_provider}, model: {args.model})..."
    )

    # Provider configuration already obtained during validation

    # Initialize model using the provider map with proper API key
    # Create a temporary config object for validation
    temp_config = type("TempConfig", (), {"llm_provider": args.llm_provider})()
    model_class = validate_and_get_model_class(temp_config)

    if args.llm_provider == "huggingface":
        model = model_class(model=args.model, timeout=30000)
    elif args.llm_provider == "google":
        # Google models use google_api_key parameter
        model = model_class(
            model=args.model, google_api_key=provider_config["api_key"], timeout=30000
        )
    else:
        # Other providers use api_key parameter
        model = model_class(
            model=args.model, api_key=provider_config["api_key"], timeout=30000
        )

    # Determine output directory
    output_dir = args.output_dir or os.path.dirname(args.config)
    os.makedirs(output_dir, exist_ok=True)
    log_context.info(f"📁 Output directory: {output_dir}")

    # Initialize generator with model and output_dir
    log_context.info("🔧 Initializing task graph generator...")
    if interactable_with_user:
        log_context.info(
            "👤 Interactive task editor is ENABLED - you will be able to edit tasks"
        )
    else:
        log_context.info(
            "🚫 Interactive task editor is DISABLED - tasks will be generated automatically"
        )

    generator = Generator(
        config,
        model,
        output_dir,
        allow_nested_graph=allow_nested_graph,
        interactable_with_user=interactable_with_user,
    )

    # Generate task graph
    log_context.info("🎯 Generating task graph...")
    task_graph = generator.generate()
    log_context.info("✅ Task graph generated successfully")

    # Save the generated task graph
    log_context.info("💾 Saving task graph...")
    taskgraph_filepath = generator.save_task_graph(task_graph)
    log_context.info(f"📄 Task graph saved to {taskgraph_filepath}")

    # Build RAG if specified
    if "rag_docs" in config:
        log_context.info("🔍 Building RAG system...")
        build_rag(os.path.dirname(args.config), config["rag_docs"])
        log_context.info("✅ RAG system built successfully")

    # Build database if specified
    if "database" in config:
        log_context.info("🗄️ Building database...")
        build_database(config["database"])
        log_context.info("✅ Database built successfully")

    elapsed_time = time.time() - start_time
    log_context.info(
        f"🎉 Task graph generation completed in {elapsed_time:.1f} seconds"
    )


if __name__ == "__main__":
    main()
