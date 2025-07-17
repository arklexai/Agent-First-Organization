# ⚙️ Configuration Guide

Complete guide to configuring Arklex AI agents for different environments and use cases.

## 📋 Table of Contents

- [Environment Variables](#-environment-variables)
- [Agent Configuration](#-agent-configuration)
- [Worker Configuration](#-worker-configuration)
- [Tool Configuration](#-tool-configuration)
- [Security Configuration](#-security-configuration)
- [Performance Configuration](#-performance-configuration)
- [Environment-Specific Configs](#-environment-specific-configs)

## 🔧 Environment Variables

### Required Variables

| **Variable** | **Description** | **Example** | **Required** |
|--------------|-----------------|-------------|--------------|
| `OPENAI_API_KEY` | OpenAI API key | `sk-...` | Yes (if using OpenAI) |
| `ANTHROPIC_API_KEY` | Anthropic API key | `sk-ant-...` | Yes (if using Anthropic) |
| `GOOGLE_API_KEY` | Google Gemini API key | `AIza...` | Yes (if using Gemini) |
| `MISTRAL_API_KEY` | Mistral API key | `...` | Yes (if using Mistral) |

### Optional Variables

#### Database Configuration

| **Variable** | **Description** | **Default** |
|--------------|-----------------|-------------|
| `MILVUS_URI` | Milvus connection URI | `localhost:19530` |
| `MILVUS_USERNAME` | Milvus username | `None` |
| `MILVUS_PASSWORD` | Milvus password | `None` |
| `MYSQL_USERNAME` | MySQL username | `None` |
| `MYSQL_PASSWORD` | MySQL password | `None` |
| `MYSQL_HOSTNAME` | MySQL hostname | `localhost` |
| `MYSQL_PORT` | MySQL port | `3306` |
| `MYSQL_DB_NAME` | MySQL database name | `arklex_db` |

#### External Services

| **Variable** | **Description** | **Default** |
|--------------|-----------------|-------------|
| `TAVILY_API_KEY` | Tavily search API key | `None` |
| `SHOPIFY_API_KEY` | Shopify API key | `None` |
| `HUBSPOT_API_KEY` | HubSpot API key | `None` |
| `GOOGLE_CALENDAR_CREDENTIALS` | Google Calendar credentials file | `None` |

#### Production Settings

| **Variable** | **Description** | **Default** |
|--------------|-----------------|-------------|
| `JWT_SECRET` | JWT secret key | `None` |
| `RATE_LIMIT_PER_MINUTE` | Rate limit per minute | `100` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `API_HOST` | API server host | `0.0.0.0` |
| `API_PORT` | API server port | `8000` |

### Environment File Example

```env
# =============================================================================
# REQUIRED: Choose at least one LLM provider
# =============================================================================

# OpenAI (recommended for production)
OPENAI_API_KEY=sk-your-openai-key-here
OPENAI_ORG_ID=org-your-org-id-here

# Anthropic (alternative)
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here

# Google Gemini (alternative)
GOOGLE_API_KEY=AIza-your-gemini-key-here

# Mistral (alternative)
MISTRAL_API_KEY=your-mistral-key-here

# =============================================================================
# OPTIONAL: Enhanced functionality
# =============================================================================

# Vector Database (Milvus)
MILVUS_URI=milvus://localhost:19530
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

# External Integrations
SHOPIFY_API_KEY=your_shopify_key_here
HUBSPOT_API_KEY=your_hubspot_key_here
GOOGLE_CALENDAR_CREDENTIALS=path/to/credentials.json

# =============================================================================
# OPTIONAL: Production settings
# =============================================================================

# Security
JWT_SECRET=your_super_secure_jwt_secret_here
RATE_LIMIT_PER_MINUTE=100
RATE_LIMIT_BURST_SIZE=20

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE=/var/log/arklex/app.log

# API Server
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Monitoring
METRICS_ENABLED=true
HEALTH_CHECK_ENABLED=true
PROMETHEUS_ENABLED=true
```

## 🎯 Agent Configuration

### Basic Configuration Schema

```json
{
  "name": "Agent Name",
  "description": "Agent description",
  "version": "1.0.0",
  "environment": "development",
  "orchestrator": {
    "llm_provider": "openai",
    "model": "gpt-4o",
    "temperature": 0.7,
    "max_tokens": 1000,
    "timeout": 30,
    "retry_attempts": 3,
    "fallback_providers": ["anthropic", "gemini"]
  },
  "workers": {
    "rag_worker": {
      "enabled": true,
      "vector_db": "milvus",
      "collection_name": "documents"
    },
    "database_worker": {
      "enabled": true,
      "connection_string": "mysql://user:pass@localhost/db"
    }
  },
  "tools": [
    "calculator_tool",
    "web_search_tool"
  ],
  "middleware": [
    "logging_middleware",
    "rate_limit_middleware"
  ]
}
```

### Orchestrator Configuration

| **Parameter** | **Type** | **Default** | **Description** |
|---------------|----------|-------------|-----------------|
| `llm_provider` | `str` | `"openai"` | LLM provider (openai, anthropic, gemini, mistral) |
| `model` | `str` | `"gpt-4o"` | Model name for the specified provider |
| `temperature` | `float` | `0.7` | Sampling temperature (0.0 to 2.0) |
| `max_tokens` | `int` | `1000` | Maximum tokens in response |
| `timeout` | `int` | `30` | Request timeout in seconds |
| `retry_attempts` | `int` | `3` | Number of retry attempts |
| `fallback_providers` | `list` | `[]` | List of fallback providers |

### Model-Specific Configuration

#### OpenAI Models

```json
{
  "orchestrator": {
    "llm_provider": "openai",
    "model": "gpt-4o",
    "temperature": 0.7,
    "max_tokens": 1000,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
  }
}
```

#### Anthropic Models

```json
{
  "orchestrator": {
    "llm_provider": "anthropic",
    "model": "claude-3-5-sonnet-20241022",
    "temperature": 0.7,
    "max_tokens": 1000,
    "top_p": 1.0,
    "top_k": 40
  }
}
```

#### Google Gemini Models

```json
{
  "orchestrator": {
    "llm_provider": "gemini",
    "model": "gemini-1.5-pro",
    "temperature": 0.7,
    "max_tokens": 1000,
    "top_p": 1.0,
    "top_k": 40
  }
}
```

## 🔧 Worker Configuration

### RAG Worker Configuration

```json
{
  "rag_worker": {
    "enabled": true,
    "vector_db": "milvus",
    "collection_name": "documents",
    "embedding_model": "text-embedding-ada-002",
    "top_k": 5,
    "similarity_threshold": 0.7,
    "cache_enabled": true,
    "cache_ttl": 3600,
    "batch_size": 10,
    "max_concurrent_requests": 5
  }
}
```

#### Vector Database Options

| **Database** | **Configuration** | **Example** |
|--------------|-------------------|-------------|
| **Milvus** | `"vector_db": "milvus"` | Default, recommended for production |
| **Pinecone** | `"vector_db": "pinecone"` | Cloud-hosted vector database |
| **Weaviate** | `"vector_db": "weaviate"` | Open-source vector database |
| **FAISS** | `"vector_db": "faiss"` | Local vector database |

#### Embedding Models

| **Model** | **Provider** | **Dimensions** | **Use Case** |
|-----------|--------------|----------------|--------------|
| `text-embedding-ada-002` | OpenAI | 1536 | General purpose |
| `text-embedding-3-small` | OpenAI | 1536 | Faster, cheaper |
| `text-embedding-3-large` | OpenAI | 3072 | Higher quality |
| `claude-3-sonnet-20240229` | Anthropic | 4096 | High quality |

### Database Worker Configuration

```json
{
  "database_worker": {
    "enabled": true,
    "connection_string": "mysql://user:pass@localhost:3306/arklex_db",
    "pool_size": 10,
    "max_overflow": 20,
    "echo": false,
    "ssl_mode": "require",
    "connection_timeout": 10,
    "query_timeout": 30
  }
}
```

#### Database Connection Strings

| **Database** | **Connection String Format** |
|--------------|------------------------------|
| **MySQL** | `mysql://user:pass@host:port/db` |
| **PostgreSQL** | `postgresql://user:pass@host:port/db` |
| **SQLite** | `sqlite:///path/to/database.db` |

### Browser Worker Configuration

```json
{
  "browser_worker": {
    "enabled": false,
    "headless": true,
    "timeout": 30,
    "user_agent": "Mozilla/5.0 (compatible; ArklexBot/1.0)",
    "proxy": "http://proxy:8080",
    "viewport_width": 1920,
    "viewport_height": 1080,
    "max_concurrent_browsers": 3
  }
}
```

## 🛠️ Tool Configuration

### Available Tools

| **Tool** | **Configuration** | **Required Keys** |
|----------|-------------------|-------------------|
| `calculator_tool` | No configuration needed | None |
| `web_search_tool` | Search API configuration | `TAVILY_API_KEY` |
| `shopify_tool` | Shopify API configuration | `SHOPIFY_API_KEY` |
| `hubspot_tool` | HubSpot API configuration | `HUBSPOT_API_KEY` |
| `google_calendar_tool` | Google Calendar OAuth | `GOOGLE_CALENDAR_CREDENTIALS` |
| `email_tool` | SMTP configuration | SMTP settings |

### Tool-Specific Configuration

#### Web Search Tool

```json
{
  "tools": [
    {
      "name": "web_search_tool",
      "config": {
        "api_key": "your_tavily_key",
        "search_depth": "basic",
        "include_domains": ["example.com"],
        "exclude_domains": ["spam.com"],
        "max_results": 5
      }
    }
  ]
}
```

#### Shopify Tool

```json
{
  "tools": [
    {
      "name": "shopify_tool",
      "config": {
        "api_key": "your_shopify_key",
        "store_url": "your-store.myshopify.com",
        "api_version": "2024-01",
        "webhook_secret": "your_webhook_secret"
      }
    }
  ]
}
```

#### HubSpot Tool

```json
{
  "tools": [
    {
      "name": "hubspot_tool",
      "config": {
        "api_key": "your_hubspot_key",
        "portal_id": "your_portal_id",
        "rate_limit": 100
      }
    }
  ]
}
```

## 🔐 Security Configuration

### Authentication Configuration

```json
{
  "security": {
    "authentication": "jwt",
    "jwt_secret": "your_jwt_secret",
    "jwt_algorithm": "HS256",
    "jwt_expiration": 3600,
    "api_keys": {
      "key1": "user1",
      "key2": "user2"
    }
  }
}
```

### Rate Limiting Configuration

```json
{
  "security": {
    "rate_limiting": {
      "enabled": true,
      "requests_per_minute": 100,
      "burst_size": 20,
      "storage": "redis",
      "redis_url": "redis://localhost:6379/0"
    }
  }
}
```

### CORS Configuration

```json
{
  "security": {
    "cors": {
      "enabled": true,
      "origins": ["https://yourdomain.com"],
      "methods": ["GET", "POST", "PUT", "DELETE"],
      "headers": ["*"],
      "credentials": true
    }
  }
}
```

### Input Validation Configuration

```json
{
  "security": {
    "input_validation": {
      "enabled": true,
      "max_query_length": 10000,
      "allowed_file_types": ["txt", "pdf", "docx"],
      "max_file_size": 10485760,
      "sanitize_html": true
    }
  }
}
```

## ⚡ Performance Configuration

### Caching Configuration

```json
{
  "performance": {
    "caching": {
      "enabled": true,
      "backend": "redis",
      "redis_url": "redis://localhost:6379/0",
      "default_ttl": 3600,
      "max_size": 1000,
      "cache_rag_results": true,
      "cache_llm_responses": true
    }
  }
}
```

### Concurrency Configuration

```json
{
  "performance": {
    "concurrency": {
      "max_workers": 4,
      "max_concurrent_requests": 10,
      "worker_timeout": 30,
      "queue_size": 100
    }
  }
}
```

### Auto-scaling Configuration

```json
{
  "performance": {
    "auto_scaling": {
      "enabled": true,
      "min_workers": 2,
      "max_workers": 10,
      "scale_up_threshold": 0.8,
      "scale_down_threshold": 0.2,
      "cooldown_period": 300,
      "metrics": ["cpu", "memory", "response_time"]
    }
  }
}
```

## 🌍 Environment-Specific Configs

### Development Configuration

```json
{
  "name": "Development Agent",
  "environment": "development",
  "orchestrator": {
    "llm_provider": "openai",
    "model": "gpt-4o-mini",
    "temperature": 0.9,
    "max_tokens": 500,
    "debug": true
  },
  "workers": {
    "rag_worker": {
      "enabled": false
    },
    "database_worker": {
      "enabled": false
    }
  },
  "tools": [],
  "middleware": ["logging_middleware"],
  "logging": {
    "level": "DEBUG",
    "format": "text"
  }
}
```

### Staging Configuration

```json
{
  "name": "Staging Agent",
  "environment": "staging",
  "orchestrator": {
    "llm_provider": "openai",
    "model": "gpt-4o",
    "temperature": 0.7,
    "max_tokens": 1000,
    "retry_attempts": 3
  },
  "workers": {
    "rag_worker": {
      "enabled": true,
      "vector_db": "milvus",
      "collection_name": "staging_documents"
    },
    "database_worker": {
      "enabled": true,
      "connection_string": "mysql://user:pass@staging-db:3306/arklex_staging"
    }
  },
  "tools": ["calculator_tool"],
  "middleware": [
    "logging_middleware",
    "rate_limit_middleware"
  ],
  "security": {
    "authentication": "jwt",
    "rate_limiting": {
      "enabled": true,
      "requests_per_minute": 50
    }
  }
}
```

### Production Configuration

```json
{
  "name": "Production Agent",
  "environment": "production",
  "orchestrator": {
    "llm_provider": "openai",
    "model": "gpt-4o",
    "temperature": 0.7,
    "max_tokens": 1000,
    "timeout": 30,
    "retry_attempts": 3,
    "fallback_providers": ["anthropic", "gemini"]
  },
  "workers": {
    "rag_worker": {
      "enabled": true,
      "vector_db": "milvus",
      "collection_name": "production_documents",
      "cache_enabled": true,
      "cache_ttl": 3600
    },
    "database_worker": {
      "enabled": true,
      "connection_string": "mysql://user:pass@prod-db:3306/arklex_production",
      "pool_size": 20,
      "max_overflow": 40
    }
  },
  "tools": [
    "calculator_tool",
    "web_search_tool"
  ],
  "middleware": [
    "logging_middleware",
    "rate_limit_middleware",
    "auth_middleware",
    "monitoring_middleware"
  ],
  "security": {
    "authentication": "jwt",
    "rate_limiting": {
      "enabled": true,
      "requests_per_minute": 100,
      "burst_size": 20
    },
    "cors": {
      "enabled": true,
      "origins": ["https://yourdomain.com"]
    }
  },
  "performance": {
    "caching": {
      "enabled": true,
      "backend": "redis",
      "default_ttl": 3600
    },
    "auto_scaling": {
      "enabled": true,
      "min_workers": 2,
      "max_workers": 10
    }
  },
  "monitoring": {
    "metrics_enabled": true,
    "health_checks": true,
    "logging": {
      "level": "INFO",
      "format": "json"
    }
  }
}
```

## 🔄 Configuration Management

### Environment-Specific Files

```bash
# Development
config.dev.json

# Staging
config.staging.json

# Production
config.prod.json

# Load based on environment
export ENVIRONMENT=production
python create.py --config config.${ENVIRONMENT}.json --llm_provider openai --model gpt-4o

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

### Configuration Validation

```python
# Validate configuration
from arklex.utils.config import validate_config

config = load_config("config.json")
errors = validate_config(config)

if errors:
    print("Configuration errors:")
    for error in errors:
        print(f"  - {error}")
    exit(1)
```

### Configuration Templates

Create reusable configuration templates:

```json
{
  "template": "customer_service",
  "description": "Customer service agent template",
  "orchestrator": {
    "llm_provider": "openai",
    "model": "gpt-4o",
    "temperature": 0.7
  },
  "workers": {
    "rag_worker": {
      "enabled": true,
      "vector_db": "milvus"
    },
    "database_worker": {
      "enabled": true
    }
  },
  "tools": ["calculator_tool"],
  "middleware": ["logging_middleware", "rate_limit_middleware"]
}
```

---

For more information on specific configuration options, see the [API Reference](API.md) and [Deployment Guide](DEPLOYMENT.md).
