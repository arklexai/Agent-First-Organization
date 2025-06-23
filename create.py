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
import tempfile
import zipfile
import time
from typing import Any, Dict, List, Optional, Set

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

from arklex.utils.logging_utils import LogContext
from arklex.orchestrator.generator.generator import Generator
from arklex.env.tools.RAG.build_rag import build_rag
from arklex.env.tools.database.build_database import build_database
from arklex.utils.model_config import MODEL
from arklex.utils.model_provider_config import PROVIDER_MAP
from arklex.utils.loader import Loader

log_context = LogContext(__name__)
load_dotenv()


def generate_taskgraph(args: argparse.Namespace) -> None:
    """Generate a task graph based on the provided configuration.

    This function initializes a language model, loads the configuration, and uses the Generator
    to create a task graph. It then saves the task graph to a file and updates it with API URLs.

    Args:
        args (argparse.Namespace): Command-line arguments containing configuration and output settings.
    """
    model = PROVIDER_MAP.get(MODEL["llm_provider"], ChatOpenAI)(
        model=MODEL["model_type_or_path"], timeout=30000
    )
    config: Dict[str, Any] = json.load(open(args.config))
    generator = Generator(config, model, args.output_dir)
    taskgraph = generator.generate()
    taskgraph_filepath: str = generator.save_task_graph(taskgraph)
    # Update the task graph with the API URLs
    task_graph: Dict[str, Any] = json.load(
        open(os.path.join(os.path.dirname(__file__), taskgraph_filepath))
    )
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
    config: Dict[str, Any] = json.load(open(args.config))
    workers: List[Dict[str, Any]] = config["workers"]
    worker_names: Set[str] = set([worker["name"] for worker in workers])

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
    config: Dict[str, Any], document_dir: Optional[str] = None
) -> List[Dict[str, str]]:
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
                                    log_context.info(
                                        f"    📦 Extracting ZIP archive..."
                                    )
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
                                    log_context.info(
                                        f"    📄 Processing single file..."
                                    )
                                    all_docs.extend(
                                        loader.to_crawled_local_objs([source])
                                    )
                                    total_docs_processed += 1
                                    log_context.info(
                                        f"    ✅ File processed successfully"
                                    )
                            elif os.path.isdir(source):
                                log_context.info(
                                    f"    📁 Processing directory contents..."
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
                            log_context.info(f"    📝 Processing text content...")
                            all_docs.extend(loader.to_crawled_text([source]))
                            total_docs_processed += 1
                            log_context.info(f"    ✅ Text content processed")
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


def main():
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
    args = parser.parse_args()

    # Handle nested graph flag
    allow_nested_graph = args.allow_nested_graph and not args.no_nested_graph

    # Set up logging
    log_context.setLevel(getattr(logging, args.log_level.upper()))

    log_context.info("🚀 Starting task graph generation...")
    start_time = time.time()

    # Load config
    log_context.info(f"📋 Loading configuration from {args.config}")
    with open(args.config, "r") as f:
        config = json.load(f)

    # Load documents
    log_context.info("📚 Loading and processing documents...")
    documents = load_documents(config)
    log_context.info(f"📄 Loaded {len(documents)} documents successfully")

    # Instantiate model
    log_context.info("🤖 Initializing language model...")
    model = PROVIDER_MAP.get(MODEL["llm_provider"], ChatOpenAI)(
        model=MODEL["model_type_or_path"], timeout=30000
    )

    # Determine output directory
    output_dir = args.output_dir or os.path.dirname(args.config)
    os.makedirs(output_dir, exist_ok=True)
    log_context.info(f"📁 Output directory: {output_dir}")

    # Initialize generator with model and output_dir
    log_context.info("🔧 Initializing task graph generator...")
    generator = Generator(
        config, model, output_dir, allow_nested_graph=allow_nested_graph
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
