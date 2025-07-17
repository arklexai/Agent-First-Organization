# Contributing to Arklex AI

Thank you for your interest in contributing to Arklex AI! This document provides guidelines and information for contributors to help make the contribution process smooth and effective.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style and Standards](#code-style-and-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Issue Reporting](#issue-reporting)
- [Community Guidelines](#community-guidelines)

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- A GitHub account
- API keys for LLM providers (OpenAI, Anthropic, etc.)

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:

   ```bash
   git clone https://github.com/YOUR_USERNAME/Agent-First-Organization.git
   cd Agent-First-Organization
   ```

3. Add the upstream repository:

   ```bash
   git remote add upstream https://github.com/arklexai/Agent-First-Organization.git
   ```

## Development Setup

### 1. Create a Virtual Environment

```bash
# Using venv (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n arklex-dev python=3.10
conda activate arklex-dev
```

### 2. Install Dependencies

```bash
# Install the package in development mode with all optional dependencies
pip install -e ".[milvus,shopify,hubspot,strict-versions]"

# Install development dependencies
pip install -r requirements-dev.txt  # if available
```

### 3. Environment Configuration

Create a `.env` file in the root directory:

```env
# Required: Choose one LLM provider
OPENAI_API_KEY=your_key_here
# OR ANTHROPIC_API_KEY=your_key_here
# OR GOOGLE_API_KEY=your_key_here

# Optional: Enhanced features
MILVUS_URI=your_milvus_uri
MYSQL_USERNAME=your_username
TAVILY_API_KEY=your_tavily_key

# Development settings
ARKLEX_TEST_ENV=local  # Use mocked responses for testing
```

### 4. Verify Installation

```bash
# Run basic tests to ensure everything is working
python -m pytest tests/ -v

# Try creating a simple agent
python create.py \
  --config ./examples/customer_service/customer_service_config.json \
  --output-dir ./test_output \
  --llm_provider openai \
  --model gpt-4o-mini

# For evaluation and testing, you can also use the model API server:
# 1. Start the model API server (defaults to OpenAI with "gpt-4o-mini" model):
python model_api.py --input-dir ./examples/customer_service

# 2. Run evaluation (in a separate terminal):
python eval.py --model_api http://127.0.0.1:8000/eval/chat \
  --config "examples/customer_service/customer_service_config.json" \
  --documents_dir "examples/customer_service" \
  --model "claude-3-haiku-20240307" \
  --llm_provider "anthropic" \
  --task "all"
```

## Code Style and Standards

### Python Code Style

We use **Ruff** for linting and formatting. The configuration is in `pyproject.toml`.

```bash
# Install pre-commit hooks (recommended)
pip install pre-commit
pre-commit install

# Or run manually
ruff check .          # Lint code
ruff format .         # Format code
```

### Code Style Guidelines

- **Line Length**: 88 characters (Black standard)
- **Indentation**: 4 spaces
- **Quotes**: Double quotes for strings
- **Type Hints**: Use type hints for all function parameters and return values
- **Docstrings**: Use Google-style docstrings for all public functions and classes

### Example Code Style

```python
from typing import Dict, List, Optional
from arklex.types import TaskGraph, Orchestrator


def create_agent(
    config_path: str,
    output_dir: str,
    llm_provider: str = "openai",
    model: str = "gpt-4o-mini",
) -> Orchestrator:
    """Create a new agent from configuration.

    Args:
        config_path: Path to the configuration file.
        output_dir: Directory to save the generated agent.
        llm_provider: LLM provider to use (openai, anthropic, gemini).
        model: Model name to use.

    Returns:
        Configured orchestrator instance.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If invalid LLM provider specified.
    """
    # Implementation here
    pass
```

### Import Organization

Imports should be organized as follows:

```python
# Standard library imports
import os
from typing import Dict, List

# Third-party imports
import fastapi
from pydantic import BaseModel

# Local imports
from arklex.orchestrator import Orchestrator
from arklex.workers import RAGWorker
```

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/orchestrator/ -v
python -m pytest tests/env/ -v
python -m pytest tests/workers/ -v

# Run with coverage
python -m pytest tests/ --cov=arklex --cov-report=html

# Run integration tests (with real LLM)
export ARKLEX_TEST_ENV=integration
python -m pytest tests/test_resources.py -v

# Run local tests (with mocked LLM)
export ARKLEX_TEST_ENV=local
python -m pytest tests/test_resources.py -v
```

### Writing Tests

#### Test Structure

```python
import pytest
from arklex.orchestrator import Orchestrator


class TestOrchestrator:
    """Test suite for Orchestrator class."""

    @pytest.fixture
    def orchestrator(self):
        """Create a test orchestrator instance."""
        return Orchestrator(
            llm_provider="openai",
            model="gpt-4o-mini",
            api_key="test_key"
        )

    def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert orchestrator.llm_provider == "openai"
        assert orchestrator.model == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_orchestrator_run(self, orchestrator):
        """Test orchestrator run method."""
        # Test implementation
        pass
```

#### Test Data

- Place test data files in `tests/data/`
- Use JSON format for configuration files
- Create separate test cases for different scenarios

#### Test Naming

- Test files: `test_*.py`
- Test classes: `Test*`
- Test methods: `test_*`
- Use descriptive names that explain what is being tested

### Adding New Tests

1. **Create test configuration** in `tests/data/`
2. **Create test cases** with expected inputs and outputs
3. **Add test case to runner** in `tests/test_resources.py`
4. **Run tests** to ensure they pass

## Documentation

### Code Documentation

- All public APIs must have docstrings
- Use Google-style docstrings
- Include type hints
- Provide usage examples for complex functions

### Documentation Structure

```bash
docs/
├── API.md              # API reference
├── ARCHITECTURE.md     # System architecture
├── CONFIGURATION.md    # Configuration guide
├── Integration/        # Integration guides
├── Example/           # Example tutorials
├── Workers/           # Worker documentation
└── Taskgraph/         # Task graph documentation
```

### Writing Documentation

- Use clear, concise language
- Include code examples
- Add diagrams for complex concepts
- Keep documentation up-to-date with code changes

## Submitting Changes

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Your Changes

- Write your code following the style guidelines
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 3. Commit Your Changes

```bash
# Stage your changes
git add .

# Commit with a descriptive message
git commit -m "feat: add new RAG worker implementation

- Add FAISS vector store support
- Implement similarity search functionality
- Add comprehensive test coverage
- Update documentation with usage examples

Closes #123"
```

### Commit Message Format

We follow the [Conventional Commits](https://www.conventionalcommits.org/) format:

- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

### 4. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:

- **Clear title** describing the change
- **Detailed description** of what was changed and why
- **Related issue** number (if applicable)
- **Screenshots** for UI changes
- **Test results** showing all tests pass

### 5. Pull Request Review

- Address any review comments
- Ensure CI/CD checks pass
- Update documentation if needed
- Respond to feedback promptly

## Issue Reporting

### Before Reporting

1. **Search existing issues** to avoid duplicates
2. **Check documentation** for solutions
3. **Try the latest version** of the code
4. **Reproduce the issue** with minimal steps

### Issue Template

When creating an issue, use the provided template and include:

- **Clear title** describing the problem
- **Detailed description** of the issue
- **Steps to reproduce** the problem
- **Expected vs actual behavior**
- **Environment information** (OS, Python version, etc.)
- **Error messages** and stack traces
- **Screenshots** (if applicable)

### Bug Reports

For bug reports, include:

```markdown
## Bug Description
Brief description of the bug.

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Environment
- OS: [e.g., macOS 14.0]
- Python: [e.g., 3.10.0]
- Arklex: [e.g., 0.1.0]
- LLM Provider: [e.g., OpenAI]

## Additional Information
Any other relevant information.
```

### Feature Requests

For feature requests, include:

```markdown
## Feature Description
Brief description of the feature.

## Use Case
Why this feature is needed and how it would be used.

## Proposed Implementation
Any ideas for how to implement this feature.

## Alternatives Considered
Other approaches that were considered.
```

## Community Guidelines

### Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please:

- **Be respectful** and inclusive in all interactions
- **Use inclusive language** in code, documentation, and discussions
- **Respect different viewpoints** and experiences
- **Focus on constructive feedback** and improvement
- **Report inappropriate behavior** to maintainers

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and community discussions
- **Discord**: Real-time chat and community support
- **Email**: <support@arklex.ai> for private matters

### Getting Help

- **Check documentation** first
- **Search existing issues** for similar problems
- **Ask in Discussions** for general questions
- **Join Discord** for real-time help
- **Create an issue** for bugs or feature requests

## Recognition

### Contributors

All contributors are recognized in:

- **GitHub Contributors** page
- **Release notes** for significant contributions
- **Documentation** acknowledgments
- **Community highlights**

### Types of Contributions

We welcome all types of contributions:

- **Code**: Bug fixes, new features, improvements
- **Documentation**: Guides, tutorials, API docs
- **Testing**: Test cases, bug reports, feedback
- **Design**: UI/UX improvements, diagrams
- **Community**: Helping others, organizing events

## Development Workflow

### Daily Development

1. **Sync with upstream**:

   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. **Create feature branch**:

   ```bash
   git checkout -b feature/your-feature
   ```

3. **Make changes** and test locally

4. **Commit and push**:

   ```bash
   git add .
   git commit -m "feat: your feature description"
   git push origin feature/your-feature
   ```

5. **Create Pull Request** and wait for review

### Release Process

1. **Version bump** in `arklex/__init__.py`
2. **Update changelog** with new features/fixes
3. **Create release tag** on GitHub
4. **Deploy to PyPI** (maintainers only)

## Questions?

If you have questions about contributing:

1. **Check this document** first
2. **Search existing issues** and discussions
3. **Ask in GitHub Discussions**
4. **Join our Discord** community
5. **Email** <support@arklex.ai>

Thank you for contributing to Arklex AI! 🚀
