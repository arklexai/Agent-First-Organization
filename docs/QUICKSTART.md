# 🚀 Quick Start Guide

Get up and running with Arklex AI in minutes! This guide will walk you through installing, configuring, and running your first intelligent agent.

## 📋 Prerequisites

Before you begin, make sure you have:

- **Python 3.10+** installed
- **8GB RAM** (recommended for production)
- **API Keys** for your chosen LLM providers
- **Basic Python knowledge**

## 🔧 Installation

### Step 1: Install Arklex AI

Choose your installation method:

```bash
# Basic installation (core functionality)
pip install arklex

# With vector database support
pip install arklex[milvus]

# With e-commerce integrations
pip install arklex[shopify]

# With CRM integrations
pip install arklex[hubspot]

# Complete installation with all features
pip install arklex[all]
```

### Step 2: Verify Installation

```bash
python -c "import arklex; print('Arklex AI installed successfully!')"
```

## ⚙️ Configuration

### Step 1: Set Up Environment Variables

Create a `.env` file in your project root:

```env
# =============================================================================
# REQUIRED: Choose at least one LLM provider
# =============================================================================

# OpenAI (recommended for production)
OPENAI_API_KEY=your_openai_key_here
OPENAI_ORG_ID=your_org_id_here  # Optional

# Anthropic (alternative)
ANTHROPIC_API_KEY=your_anthropic_key_here

# Google Gemini (alternative)
GOOGLE_API_KEY=your_gemini_key_here

# Mistral (alternative)
MISTRAL_API_KEY=your_mistral_key_here

# Hugging Face (for open-source models)
HUGGINGFACE_API_KEY=your_huggingface_key_here

# =============================================================================
# OPTIONAL: Enhanced functionality
# =============================================================================

# LangChain integration
LANGCHAIN_API_KEY=your_langchain_key_here
LANGCHAIN_PROJECT=AgentOrg
LANGCHAIN_TRACING_V2=false

# Vector Database (Milvus)
MILVUS_URI=your_milvus_uri_here
MILVUS_USERNAME=your_milvus_username
MILVUS_PASSWORD=your_milvus_password

# SQL Database (MySQL)
MYSQL_USERNAME=your_mysql_username
MYSQL_PASSWORD=your_mysql_password
MYSQL_HOSTNAME=localhost
MYSQL_PORT=3306
MYSQL_DB_NAME=arklex_db
MYSQL_CONNECTION_TIMEOUT=10

# Web Search (Tavily)
TAVILY_API_KEY=your_tavily_key_here

# =============================================================================
# OPTIONAL: Production settings
# =============================================================================

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# API Server
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Security
JWT_SECRET=your_jwt_secret_here
RATE_LIMIT_PER_MINUTE=100
```

### Step 2: Test Your Configuration

```bash
# Test API connectivity
python -c "
import os
from arklex import Orchestrator

orchestrator = Orchestrator(
    llm_provider='openai',
    model='gpt-4o-mini',
    api_key=os.getenv('OPENAI_API_KEY')
)
print('Configuration test successful!')
"
```

## 🎯 Your First Agent

### Step 1: Create Agent Configuration

Create a file called `my_first_agent_config.json`:

```json
{
  "name": "My First Agent",
  "description": "A simple agent that can answer questions",
  "version": "1.0.0",
  "orchestrator": {
    "llm_provider": "openai",
    "model": "gpt-4o-mini",
    "temperature": 0.7,
    "max_tokens": 1000
  },
  "workers": {
    "rag_worker": {
      "enabled": false
    },
    "database_worker": {
      "enabled": false
    }
  },
  "tools": []
}
```

### Step 2: Generate Agent Workflow

```bash
python create.py \
  --config my_first_agent_config.json \
  --output-dir ./my_first_agent \
  --llm_provider openai \
  --model gpt-4o-mini
```

### Step 3: Run Your Agent

```bash
python run.py \
  --input-dir ./my_first_agent \
  --llm_provider openai \
  --model gpt-4o-mini \
  --query "Hello! What can you help me with?"

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

## 🧠 Building a RAG-Powered Agent

### Step 1: Prepare Your Documents

Create a directory for your documents:

```bash
mkdir -p documents
# Add your PDF, TXT, or DOC files to this directory
```

### Step 2: Create RAG Agent Configuration

```json
{
  "name": "RAG Agent",
  "description": "Agent with document retrieval capabilities",
  "version": "1.0.0",
  "orchestrator": {
    "llm_provider": "openai",
    "model": "gpt-4o-mini",
    "temperature": 0.7,
    "max_tokens": 1000
  },
  "workers": {
    "rag_worker": {
      "enabled": true,
      "vector_db": "milvus",
      "collection_name": "my_documents"
    }
  },
  "tools": []
}
```

### Step 3: Generate and Run RAG Agent

```bash
# Generate the agent
python create.py \
  --config rag_agent_config.json \
  --output-dir ./rag_agent \
  --llm_provider openai \
  --model gpt-4o-mini

# Run with document query
python run.py \
  --input-dir ./rag_agent \
  --llm_provider openai \
  --model gpt-4o-mini \
  --query "What information do you have about machine learning?"
```

## 🔌 Adding Tools

### Step 1: Update Configuration with Tools

```json
{
  "name": "Tool-Enabled Agent",
  "description": "Agent with external tool integrations",
  "version": "1.0.0",
  "orchestrator": {
    "llm_provider": "openai",
    "model": "gpt-4o-mini",
    "temperature": 0.7,
    "max_tokens": 1000
  },
  "workers": {
    "rag_worker": {
      "enabled": true,
      "vector_db": "milvus",
      "collection_name": "documents"
    }
  },
  "tools": [
    "calculator_tool",
    "web_search_tool"
  ]
}
```

### Step 2: Run Tool-Enabled Agent

```bash
python run.py \
  --input-dir ./tool_agent \
  --llm_provider openai \
  --model gpt-4o-mini \
  --query "What's 15% of 250? Also, what's the latest news about AI?"
```

## 🌐 Deploy as API

### Step 1: Start API Server

```bash
python model_api.py --input-dir ./my_first_agent
```

### Step 2: Test API Endpoints

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test chat endpoint
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello, how are you?"}'
```

### Step 3: Access API Documentation

Open your browser and go to: `http://localhost:8000/docs`

## 🧪 Testing Your Agent

### Step 1: Run Basic Tests

```bash
python eval.py \
  --config my_first_agent_config.json \
  --input-dir ./my_first_agent \
  --test_queries "Hello,What is AI?,How do you work?"
```

### Step 2: Generate Synthetic Tests

```bash
python eval.py \
  --config my_first_agent_config.json \
  --input-dir ./my_first_agent \
  --synthetic_tests 50
```

## 🚀 Next Steps

Now that you have your first agent running, here are some next steps:

1. **Explore Examples** — Check out the [examples directory](../examples/) for more complex use cases
2. **Read the API Documentation** — See [API Reference](API.md) for detailed usage
3. **Learn About Architecture** — Understand the system design in [Architecture Guide](ARCHITECTURE.md)
4. **Deploy to Production** — Follow the [Deployment Guide](DEPLOYMENT.md)
5. **Join the Community** — Get help and share your projects on [GitHub Discussions](https://github.com/arklexai/Agent-First-Organization/discussions)

## 🔧 Troubleshooting

### Common Issues

**API Key Errors**

```bash
# Verify your API key is set correctly
export OPENAI_API_KEY="your-actual-key-here"
echo $OPENAI_API_KEY  # Should display your key
```

**Import Errors**

```bash
# Reinstall with all dependencies
pip uninstall arklex
pip install arklex[all]
```

**Memory Issues**

```bash
# Monitor memory usage
htop  # or top on macOS
```

For more detailed troubleshooting, see [Troubleshooting Guide](TROUBLESHOOTING.md).

---

**Need help?** Join our [Discord community](https://discord.gg/arklex) or check out the [GitHub Discussions](https://github.com/arklexai/Agent-First-Organization/discussions)!
