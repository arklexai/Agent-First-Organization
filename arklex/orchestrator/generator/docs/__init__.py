"""Public API for document loading and processing components in the Arklex framework.

This module provides specialized components for loading and processing various
types of documentation used in task graph generation. The components handle
different document formats and sources, ensuring consistent processing and
integration with the task generation system.

Components:

1. DocumentLoader
   - Loads documentation from various sources (files, URLs, etc.)
   - Processes different document formats (text, markdown, etc.)
   - Handles document validation and error checking
   - Manages document caching and optimization

2. DocumentProcessor
   - Processes raw document content
   - Extracts relevant information
   - Formats content for task generation
   - Handles document-specific requirements

3. DocumentValidator
   - Validates document structure and content
   - Ensures required fields are present
   - Checks for consistency and completeness
   - Provides detailed error messages

The components work together to provide a robust document handling system
that ensures high-quality input for task generation.

Usage:
    from arklex.orchestrator.generator.docs import DocumentLoader

    # Initialize document loader
    doc_loader = DocumentLoader(output_dir="output")

    # Load task documentation
    task_docs = doc_loader.load_task_document([
        "path/to/task1.md",
        "path/to/task2.md"
    ])

    # Load instruction documentation
    instruction_docs = doc_loader.load_instructions([
        "path/to/instructions.md"
    ])

    # Process documents
    processed_docs = doc_loader.process_documents(task_docs)
"""

from .document_loader import DocumentLoader
from .document_processor import DocumentProcessor

__all__ = ["DocumentLoader", "DocumentProcessor"]
