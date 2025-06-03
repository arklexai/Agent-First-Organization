"""Worker implementations for the Arklex framework.

This module contains various worker implementations that handle different types of tasks
within the Arklex framework. Each worker is specialized for a specific functionality,
such as message generation, database operations, search, and RAG (Retrieval-Augmented
Generation) tasks.

Key Components:
- Base Worker (worker.py):
  - Abstract base class for all workers
  - Common functionality and interface definition
- Message Workers:
  - Basic message generation (message_worker.py)
  - RAG-enhanced message generation (rag_message_worker.py)
- Database Workers:
  - Database operations and queries (database_worker.py)
- Search Workers:
  - Search functionality (search_worker.py)
- RAG Workers:
  - FAISS-based RAG (faiss_rag_worker.py)
  - Milvus-based RAG (milvus_rag_worker.py)
- HITL Worker:
  - Human-in-the-loop functionality (hitl_worker.py)

Key Features:
- Modular worker architecture
- Specialized task handling
- Extensible worker system
- Integration with RAG systems
- Database interaction support
- Human-in-the-loop capabilities

Usage:
    # Import specific workers as needed
    from arklex.env.workers import (
        BaseWorker,
        MessageWorker,
        RAGMessageWorker,
        DatabaseWorker,
        SearchWorker,
        FAISSRAGWorker,
        MilvusRAGWorker,
        HITLWorker
    )

    # Initialize and use workers
    worker = MessageWorker()
    response = worker.execute(message_state, **kwargs)
"""

# Note: Each worker implementation provides specialized functionality
# while adhering to the common interface defined in the base worker class.
