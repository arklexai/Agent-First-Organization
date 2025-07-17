# 🛠️ API Reference

Complete API documentation for Arklex AI. This guide covers all core components, classes, and methods.

## 📋 Table of Contents

- [Core Components](#-core-components)
- [Orchestrator](#-orchestrator)
- [TaskGraph](#-taskgraph)
- [Workers](#-workers)
- [Tools](#-tools)
- [Command Line Tools](#-command-line-tools)
- [Configuration](#-configuration)

## 🎯 Core Components

### Main Entry Points

| **Component** | **Purpose** | **Usage** |
|---------------|-------------|-----------|
| **`create.py`** | Generate agent workflows from configuration | `python create.py --config config.json` |
| **`run.py`** | Execute agent workflows with input data | `python run.py --input-dir ./agent` |
| **`model_api.py`** | Start FastAPI server for agent interactions | `python model_api.py --input-dir ./agent` |
| **`eval.py`** | Run evaluation and testing suites | `python eval.py --config config.json` |

## 🎼 Orchestrator

The main runtime for agent execution and workflow management.

### Basic Usage

```python
from arklex import Orchestrator

orchestrator = Orchestrator(
    llm_provider="openai",
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.7,
    max_tokens=1000
)
```

### Constructor Parameters

| **Parameter** | **Type** | **Default** | **Description** |
|---------------|----------|-------------|-----------------|
| `llm_provider` | `str` | `"openai"` | LLM provider (openai, anthropic, gemini, etc.) |
| `model` | `str` | `"gpt-4o"` | Model name for the specified provider |
| `api_key` | `str` | `None` | API key for the LLM provider |
| `temperature` | `float` | `0.7` | Sampling temperature (0.0 to 2.0) |
| `max_tokens` | `int` | `1000` | Maximum tokens in response |
| `debug` | `bool` | `False` | Enable debug mode for detailed logging |

### Methods

#### `add_worker(worker)`

Add a worker to the orchestrator.

```python
from arklex.workers import RAGWorker, DatabaseWorker

orchestrator.add_worker(RAGWorker(vector_db="milvus"))
orchestrator.add_worker(DatabaseWorker())
```

#### `add_tool(tool)`

Add a tool to the orchestrator.

```python
from arklex.tools import ShopifyTool, HubSpotTool

orchestrator.add_tool(ShopifyTool())
orchestrator.add_tool(HubSpotTool())
```

#### `run(task_graph, query, context=None)`

Execute a task graph with the given query.

```python
result = orchestrator.run(task_graph, query="How do I reset my password?")
print(result.response)
print(result.metadata)
```

#### `get_logs()`

Retrieve execution logs.

```python
logs = orchestrator.get_logs()
for log in logs:
    print(f"{log.timestamp}: {log.message}")
```

#### `get_metrics()`

Get performance metrics.

```python
metrics = orchestrator.get_metrics()
print(f"Total requests: {metrics.total_requests}")
print(f"Average latency: {metrics.avg_latency}ms")
```

## 📊 TaskGraph

DAG representation of agent workflows.

### Basic Usage

```python
from arklex import TaskGraph

task_graph = TaskGraph([
    {
        "id": "process_input",
        "type": "input",
        "description": "Process user input"
    },
    {
        "id": "search_docs",
        "type": "rag_worker",
        "description": "Search knowledge base",
        "dependencies": ["process_input"]
    },
    {
        "id": "generate_response",
        "type": "llm_worker",
        "description": "Generate final response",
        "dependencies": ["search_docs"]
    }
])
```

### Node Types

| **Type** | **Description** | **Use Case** |
|----------|-----------------|--------------|
| `input` | Process user input | Initial query processing |
| `llm_worker` | LLM-based processing | Text generation, analysis |
| `rag_worker` | Document retrieval | Knowledge base search |
| `database_worker` | Database operations | Data persistence, retrieval |
| `browser_worker` | Web automation | Web scraping, form filling |
| `custom_worker` | Custom logic | Domain-specific processing |

### Node Configuration

```python
{
    "id": "unique_node_id",
    "type": "worker_type",
    "description": "Human-readable description",
    "dependencies": ["node_id_1", "node_id_2"],
    "config": {
        "parameter1": "value1",
        "parameter2": "value2"
    }
}
```

## 🔧 Workers

Modular components for specific tasks.

### RAGWorker

Document retrieval and question-answering.

```python
from arklex.workers import RAGWorker

# Basic RAG worker
rag_worker = RAGWorker(
    vector_db="milvus",
    collection_name="documents"
)

# With custom configuration
rag_worker = RAGWorker(
    vector_db="milvus",
    collection_name="documents",
    embedding_model="text-embedding-ada-002",
    top_k=5,
    similarity_threshold=0.7
)
```

#### Parameters

| **Parameter** | **Type** | **Default** | **Description** |
|---------------|----------|-------------|-----------------|
| `vector_db` | `str` | `"milvus"` | Vector database type |
| `collection_name` | `str` | `"documents"` | Collection name |
| `embedding_model` | `str` | `"text-embedding-ada-002"` | Embedding model |
| `top_k` | `int` | `5` | Number of documents to retrieve |
| `similarity_threshold` | `float` | `0.7` | Minimum similarity score |

### DatabaseWorker

SQL operations and data persistence.

```python
from arklex.workers import DatabaseWorker

# Basic database worker
db_worker = DatabaseWorker(
    connection_string="mysql://user:pass@localhost/db"
)

# With custom configuration
db_worker = DatabaseWorker(
    connection_string="mysql://user:pass@localhost/db",
    pool_size=10,
    max_overflow=20,
    echo=False
)
```

#### Parameters

| **Parameter** | **Type** | **Default** | **Description** |
|---------------|----------|-------------|-----------------|
| `connection_string` | `str` | `None` | Database connection string |
| `pool_size` | `int` | `5` | Connection pool size |
| `max_overflow` | `int` | `10` | Maximum overflow connections |
| `echo` | `bool` | `False` | Enable SQL logging |

### BrowserWorker

Web automation and scraping.

```python
from arklex.workers import BrowserWorker

# Basic browser worker
browser_worker = BrowserWorker(
    headless=True,
    timeout=30
)

# With custom configuration
browser_worker = BrowserWorker(
    headless=True,
    timeout=30,
    user_agent="Custom User Agent",
    proxy="http://proxy:8080"
)
```

#### Parameters

| **Parameter** | **Type** | **Default** | **Description** |
|---------------|----------|-------------|-----------------|
| `headless` | `bool` | `True` | Run browser in headless mode |
| `timeout` | `int` | `30` | Page load timeout in seconds |
| `user_agent` | `str` | `None` | Custom user agent string |
| `proxy` | `str` | `None` | Proxy server URL |

## 🛠️ Tools

Atomic utilities for agent operations.

### Available Tools

| **Tool** | **Purpose** | **Configuration** |
|----------|-------------|-------------------|
| `ShopifyTool` | E-commerce operations | API credentials |
| `HubSpotTool` | CRM management | API credentials |
| `GoogleCalendarTool` | Scheduling | OAuth credentials |
| `CalculatorTool` | Mathematical operations | None required |
| `WebSearchTool` | Real-time information | Search API key |
| `EmailTool` | Email operations | SMTP credentials |

### Basic Usage

```python
from arklex.tools import ShopifyTool, HubSpotTool, CalculatorTool

# Add tools to orchestrator
orchestrator.add_tool(ShopifyTool(
    api_key="your_shopify_key",
    store_url="your-store.myshopify.com"
))

orchestrator.add_tool(HubSpotTool(
    api_key="your_hubspot_key"
))

orchestrator.add_tool(CalculatorTool())
```

### Custom Tools

Create custom tools by extending the base tool class:

```python
from arklex.tools import BaseTool

class CustomTool(BaseTool):
    def __init__(self, api_key=None):
        super().__init__()
        self.api_key = api_key
    
    def execute(self, input_data):
        # Your custom logic here
        result = self._process_input(input_data)
        return {
            "success": True,
            "data": result
        }
    
    def _process_input(self, data):
        # Custom processing logic
        return f"Processed: {data}"
```

## 💻 Command Line Tools

### create.py

Generate agent workflows from configuration.

```bash
python create.py \
  --config config.json \
  --output-dir ./agent \
  --llm_provider openai \
  --model gpt-4o-mini \
  --verbose
```

#### Parameters

| **Parameter** | **Description** | **Required** |
|---------------|-----------------|--------------|
| `--config` | Configuration file path | Yes |
| `--output-dir` | Output directory | Yes |
| `--llm_provider` | LLM provider | No |
| `--model` | Model name | No |
| `--verbose` | Enable verbose output | No |

### run.py

Execute agent workflows with input data.

```bash
python run.py \
  --input-dir ./agent \
  --llm_provider openai \
  --model gpt-4o \
  --query "Your question here" \
  --output-format json
```

#### Parameters

| **Parameter** | **Description** | **Required** |
|---------------|-----------------|--------------|
| `--input-dir` | Agent directory | Yes |
| `--llm_provider` | LLM provider | No |
| `--model` | Model name | No |
| `--query` | Input query | Yes |
| `--output-format` | Output format (json/text) | No |

### model_api.py

Start FastAPI server for agent interactions.

```bash
python model_api.py \
  --input-dir ./agent \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4
```

#### Parameters

| **Parameter** | **Description** | **Default** |
|---------------|-----------------|-------------|
| `--input-dir` | Agent directory | Required |
| `--host` | Server host | `0.0.0.0` |
| `--port` | Server port | `8000` |
| `--workers` | Number of workers | `4` |

### eval.py

Run evaluation and testing suites.

```bash
# First, start the model API server (defaults to OpenAI with "gpt-4o-mini" model):
python model_api.py --input-dir ./examples/customer_service

# Then run evaluation (in a separate terminal):
python eval.py --model_api http://127.0.0.1:8000/eval/chat \
  --config "examples/customer_service/customer_service_config.json" \
  --documents_dir "examples/customer_service" \
  --model "claude-3-haiku-20240307" \
  --llm_provider "anthropic" \
  --task "all"
```

#### Parameters

| **Parameter** | **Description** | **Required** |
|---------------|-----------------|--------------|
| `--model_api` | URL of the API endpoint for the dialogue model | Yes |
| `--config` | Configuration file | Yes |
| `--documents_dir` | Documents directory | Yes |
| `--model` | Model to use for evaluation | No (default: "gpt-4o-mini") |
| `--llm_provider` | LLM provider to use | No (default: "openai") |
| `--task` | Task type ("first_pass", "simulate_conv_only", "all") | No (default: "first_pass") |
| `--output-dir` | Results directory | No |

## ⚙️ Configuration

### Agent Configuration Schema

```json
{
  "name": "Agent Name",
  "description": "Agent description",
  "version": "1.0.0",
  "orchestrator": {
    "llm_provider": "openai",
    "model": "gpt-4o",
    "temperature": 0.7,
    "max_tokens": 1000,
    "debug": false
  },
  "workers": {
    "rag_worker": {
      "enabled": true,
      "vector_db": "milvus",
      "collection_name": "documents",
      "embedding_model": "text-embedding-ada-002",
      "top_k": 5,
      "similarity_threshold": 0.7
    },
    "database_worker": {
      "enabled": true,
      "connection_string": "mysql://user:pass@localhost/db",
      "pool_size": 5,
      "max_overflow": 10
    },
    "browser_worker": {
      "enabled": false,
      "headless": true,
      "timeout": 30
    }
  },
  "tools": [
    "shopify_tool",
    "hubspot_tool",
    "calculator_tool"
  ],
  "middleware": [
    "logging_middleware",
    "rate_limit_middleware"
  ]
}
```

### Environment Variables

| **Variable** | **Description** | **Required** |
|--------------|-----------------|--------------|
| `OPENAI_API_KEY` | OpenAI API key | Yes (if using OpenAI) |
| `ANTHROPIC_API_KEY` | Anthropic API key | Yes (if using Anthropic) |
| `GOOGLE_API_KEY` | Google Gemini API key | Yes (if using Gemini) |
| `MILVUS_URI` | Milvus connection URI | No |
| `MYSQL_USERNAME` | MySQL username | No |
| `MYSQL_PASSWORD` | MySQL password | No |
| `TAVILY_API_KEY` | Tavily search API key | No |

## 🔍 Error Handling

### Common Exceptions

```python
from arklex.utils.exceptions import (
    OrchestratorError,
    WorkerError,
    ToolError,
    ConfigurationError
)

try:
    result = orchestrator.run(task_graph, query)
except OrchestratorError as e:
    print(f"Orchestrator error: {e}")
except WorkerError as e:
    print(f"Worker error: {e}")
except ToolError as e:
    print(f"Tool error: {e}")
```

### Error Types

| **Exception** | **Description** | **Common Causes** |
|---------------|-----------------|-------------------|
| `OrchestratorError` | General orchestrator errors | Invalid configuration, API failures |
| `WorkerError` | Worker-specific errors | Database connection, RAG failures |
| `ToolError` | Tool execution errors | API rate limits, authentication |
| `ConfigurationError` | Configuration validation errors | Missing required fields, invalid values |

## 📊 Performance Optimization

### Caching

```python
# Enable caching
orchestrator.enable_caching()

# Configure cache settings
orchestrator.configure_cache(
    ttl=3600,  # 1 hour
    max_size=1000
)
```

### Concurrency

```python
# Configure worker concurrency
orchestrator.configure_concurrency(
    max_workers=4,
    max_concurrent_requests=10
)
```

### Monitoring

```python
# Enable performance monitoring
orchestrator.enable_monitoring()

# Get performance metrics
metrics = orchestrator.get_performance_metrics()
print(f"Average response time: {metrics.avg_response_time}ms")
print(f"Success rate: {metrics.success_rate}%")
```

---

For more detailed examples and advanced usage, see the [Examples Guide](EXAMPLES.md) and [Architecture Guide](ARCHITECTURE.md).
