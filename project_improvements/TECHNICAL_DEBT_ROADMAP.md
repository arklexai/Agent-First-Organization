# Technical Debt Roadmap

## High Priority

### 1. Complete Type Annotations

- **Files**: `arklex/memory/core.py`, `arklex/env/tools/RAG/retrievers/retriever_document.py`
- **Impact**: Improves code maintainability and IDE support
- **Effort**: Medium

### 2. Implement Missing Features

- **Multi-tag support for document retrieval** (`arklex/env/tools/RAG/retrievers/milvus_retriever.py`)
- **Token counting for RAG processing** (multiple files)
- **Chinese speech prompts** (`arklex/env/workers/message_worker.py`)
- **Type validation in document processing** (`arklex/env/tools/RAG/build_rag.py`)

### 3. Clean Up Inactive Modules

- **Remove or complete**: Chat client/server utilities (`arklex/env/workers/utils/chat_client.py`, `arklex/env/workers/utils/chat_server.py`)
- **Evaluate**: Shopify integration modules (multiple files marked as inactive)
- **Document**: Purpose and status of inactive modules
- **Address**: Multiple TODO comments in inactive modules

## Medium Priority

### 4. Code Organization

- **Consolidate**: Similar utility functions across modules
- **Extract**: Common patterns into shared utilities
- **Standardize**: Error handling and logging patterns
- **Refactor**: UI components to separate business logic from rendering (multiple TODO items)
- **Implement**: Missing type annotations in core modules

### 5. Performance Optimization

- **Database connections**: Implement connection pooling
- **Caching**: Add Redis caching for frequently accessed data
- **Async operations**: Optimize I/O operations
- **Token counting**: Implement token counting for RAG processing (multiple files)
- **Memory management**: Optimize memory usage in large operations

## Low Priority

### 6. Documentation Improvements

- **API documentation**: Add OpenAPI/Swagger docs
- **Architecture diagrams**: Update with current implementation
- **Tutorial videos**: Create more comprehensive guides
- **Security documentation**: Add comprehensive security guidelines
- **Testing documentation**: Document testing strategies and coverage gaps

### 7. Developer Experience

- **IDE configuration**: Add editor config files
- **Development scripts**: Automate common tasks
- **Debugging tools**: Add better debugging utilities
- **Test coverage**: Improve test coverage for UI components and error paths
- **Performance monitoring**: Add performance testing and monitoring tools

## Success Metrics

- [ ] Zero TODO comments in production code
- [ ] 100% type annotation coverage
- [ ] <100K lines of active code
- [ ] All inactive modules either completed or removed
- [ ] Performance benchmarks established
- [ ] 100% test coverage for active code
- [ ] All security vulnerabilities addressed
- [ ] UI components fully tested (no placeholder tests)
- [ ] Error paths thoroughly tested
- [ ] Performance testing framework implemented
