# 🔧 Troubleshooting Guide

Common issues, error messages, and solutions for Arklex AI deployment and usage.

## 📋 Table of Contents

- [Installation Issues](#-installation-issues)
- [Configuration Problems](#-configuration-problems)
- [API Key Errors](#-api-key-errors)
- [Database Issues](#-database-issues)
- [Performance Problems](#-performance-problems)
- [Memory Issues](#-memory-issues)
- [Network Problems](#-network-problems)
- [Production Issues](#-production-issues)

## 🔧 Installation Issues

### Python Version Problems

**Error:** `SyntaxError: invalid syntax` or `ModuleNotFoundError`

**Solution:**

```bash
# Check Python version
python --version  # Should be 3.10+

# If using older version, upgrade Python
# On macOS with Homebrew:
brew install python@3.10

# On Ubuntu/Debian:
sudo apt update
sudo apt install python3.10 python3.10-pip

# Create virtual environment with correct Python version
python3.10 -m venv venv
source venv/bin/activate
```

### Package Installation Failures

**Error:** `pip install arklex` fails with compilation errors

**Solution:**

```bash
# Install system dependencies first
# On Ubuntu/Debian:
sudo apt-get update
sudo apt-get install -y gcc g++ python3-dev

# On macOS:
xcode-select --install

# On Windows:
# Install Visual Studio Build Tools

# Then install arklex
pip install arklex

# If still failing, try with specific version
pip install arklex==1.0.0
```

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'arklex'`

**Solution:**

```bash
# Verify installation
pip list | grep arklex

# Reinstall if needed
pip uninstall arklex
pip install arklex

# Check Python path
python -c "import sys; print(sys.path)"

# Activate virtual environment if using one
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

## ⚙️ Configuration Problems

### Environment Variables Not Loading

**Error:** `KeyError: 'OPENAI_API_KEY'` or similar

**Solution:**

```bash
# Check if .env file exists
ls -la .env

# Verify environment variables are loaded
python -c "import os; print(os.getenv('OPENAI_API_KEY'))"

# Load environment variables manually
export OPENAI_API_KEY="your-key-here"

# Or use python-dotenv
pip install python-dotenv
```

### Configuration File Errors

**Error:** `ConfigurationError: Invalid configuration`

**Solution:**

```bash
# Validate JSON syntax
python -m json.tool config.json

# Check required fields
cat config.json | jq '.orchestrator.llm_provider'

# Common issues:
# - Missing required fields
# - Invalid JSON syntax
# - Wrong data types
```

## 🔑 API Key Errors

### OpenAI API Issues

**Error:** `openai.AuthenticationError: Incorrect API key provided`

**Solution:**

```bash
# Verify API key format
echo $OPENAI_API_KEY | head -c 10  # Should start with "sk-"

# Test API connectivity
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/models

# Check API key permissions
# Ensure key has access to required models (gpt-4o, text-embedding-ada-002)

# Verify billing status
# Check OpenAI dashboard for billing issues
```

### Rate Limiting

**Error:** `openai.RateLimitError: Rate limit exceeded`

**Solution:**

```python
# Implement retry logic with exponential backoff
import time
import random

def api_call_with_retry(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return func()
        except RateLimitError:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait_time)
            else:
                raise
```

### Quota Exceeded

**Error:** `openai.QuotaExceededError: You exceeded your current quota`

**Solution:**

```bash
# Check usage in OpenAI dashboard
# Add billing information
# Consider using different models or providers

# Switch to alternative provider
export ANTHROPIC_API_KEY="your-anthropic-key"
# Update config to use anthropic
```

## 🗄️ Database Issues

### MySQL Connection Problems

**Error:** `OperationalError: (2003, "Can't connect to MySQL server")`

**Solution:**

```bash
# Test MySQL connection
mysql -u username -p -h hostname -P port database_name

# Check MySQL service status
sudo systemctl status mysql

# Verify connection parameters
echo $MYSQL_HOSTNAME
echo $MYSQL_PORT
echo $MYSQL_USERNAME

# Common issues:
# - Wrong host/port
# - Firewall blocking connection
# - MySQL not running
# - Wrong credentials
```

### Milvus Connection Issues

**Error:** `ConnectionError: Failed to connect to Milvus`

**Solution:**

```bash
# Test Milvus connection
curl http://localhost:19530/health

# Check Milvus service status
docker ps | grep milvus

# Verify connection parameters
echo $MILVUS_URI
echo $MILVUS_USERNAME

# Restart Milvus if needed
docker-compose restart milvus
```

### Database Performance Issues

**Error:** Slow queries or timeouts

**Solution:**

```sql
-- Check slow queries
SHOW PROCESSLIST;

-- Optimize database
OPTIMIZE TABLE your_table;

-- Add indexes
CREATE INDEX idx_column_name ON table_name(column_name);

-- Increase connection pool
# In config:
"pool_size": 20,
"max_overflow": 40
```

## ⚡ Performance Problems

### Slow Response Times

**Symptoms:** High latency, timeouts

**Diagnosis:**

```python
# Enable debug logging
orchestrator.debug = True

# Check response times
import time
start_time = time.time()
result = orchestrator.run(task_graph, query)
end_time = time.time()
print(f"Response time: {end_time - start_time:.2f}s")

# Profile individual components
# Check LLM API response times
# Monitor database query performance
# Verify network latency
```

**Solutions:**

```python
# Enable caching
orchestrator.enable_caching()

# Use smaller models for development
orchestrator.model = "gpt-4o-mini"

# Optimize worker configuration
rag_worker.configure(
    batch_size=10,
    max_concurrent_requests=5
)

# Add connection pooling
db_worker.configure(
    pool_size=10,
    max_overflow=20
)
```

### High Memory Usage

**Symptoms:** Out of memory errors, slow performance

**Diagnosis:**

```bash
# Monitor memory usage
htop  # or top on macOS
free -h
ps aux | grep python

# Check for memory leaks
# Monitor garbage collection
```

**Solutions:**

```python
# Limit concurrent workers
orchestrator.max_concurrent_workers = 4

# Enable garbage collection
import gc
gc.enable()

# Monitor memory usage
import psutil
print(f"Memory usage: {psutil.virtual_memory().percent}%")

# Optimize data structures
# Use generators instead of lists
# Clear unused variables
```

### CPU Bottlenecks

**Symptoms:** High CPU usage, slow processing

**Diagnosis:**

```bash
# Monitor CPU usage
top
htop
iostat

# Profile Python code
python -m cProfile -o profile.stats your_script.py
```

**Solutions:**

```python
# Use async operations
import asyncio

# Implement caching
orchestrator.enable_caching()

# Optimize algorithms
# Use more efficient data structures
# Consider parallel processing
```

## 🌐 Network Problems

### Connection Timeouts

**Error:** `requests.exceptions.Timeout`

**Solution:**

```python
# Increase timeout values
orchestrator.timeout = 60

# Implement retry logic
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

session = requests.Session()
retry = Retry(connect=3, backoff_factor=0.5)
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)
```

### DNS Resolution Issues

**Error:** `socket.gaierror: [Errno -2] Name or service not known`

**Solution:**

```bash
# Test DNS resolution
nslookup api.openai.com
ping api.openai.com

# Check /etc/hosts file
cat /etc/hosts

# Use IP addresses if needed
# Or configure DNS servers
```

### Proxy Issues

**Error:** Connection failures through corporate proxy

**Solution:**

```bash
# Set proxy environment variables
export HTTP_PROXY="http://proxy.company.com:8080"
export HTTPS_PROXY="http://proxy.company.com:8080"
export NO_PROXY="localhost,127.0.0.1"

# Or configure in Python
import os
os.environ['HTTP_PROXY'] = 'http://proxy.company.com:8080'
os.environ['HTTPS_PROXY'] = 'http://proxy.company.com:8080'
```

## 🏭 Production Issues

### Service Crashes

**Symptoms:** Service stops responding, 500 errors

**Diagnosis:**

```bash
# Check service logs
tail -f /var/log/arklex/app.log

# Check system resources
htop
df -h
free -h

# Check for memory leaks
# Monitor error rates
# Check external service status
```

**Solutions:**

```python
# Implement health checks
@app.get("/health")
async def health_check():
    try:
        # Test database connection
        # Test LLM API
        # Check memory usage
        return {"status": "healthy"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

# Add circuit breakers
# Implement graceful degradation
# Set up monitoring and alerting
```

### High Error Rates

**Symptoms:** Many failed requests, poor user experience

**Diagnosis:**

```python
# Monitor error rates
error_count = 0
total_requests = 0

def track_errors():
    global error_count, total_requests
    total_requests += 1
    # Track in monitoring system

# Check error logs
# Analyze error patterns
# Monitor external service health
```

**Solutions:**

```python
# Implement retry logic
# Add fallback providers
# Use circuit breakers
# Implement graceful degradation

# Example fallback logic
try:
    result = openai_client.chat.completions.create(...)
except Exception as e:
    # Fallback to different provider
    result = anthropic_client.messages.create(...)
```

### Scaling Issues

**Symptoms:** High latency under load, timeouts

**Diagnosis:**

```bash
# Monitor load
htop
netstat -tulpn | grep :8000

# Check queue lengths
# Monitor response times
# Analyze bottlenecks
```

**Solutions:**

```python
# Implement load balancing
# Add more workers
# Use caching
# Optimize database queries

# Auto-scaling configuration
orchestrator.configure_scaling(
    min_workers=2,
    max_workers=10,
    scale_up_threshold=0.8,
    scale_down_threshold=0.2
)
```

## 🔍 Debugging Tools

### Enable Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable orchestrator debug mode
orchestrator.debug = True

# Enable worker debug mode
rag_worker.debug = True
db_worker.debug = True
```

### Performance Profiling

```python
# Profile code execution
import cProfile
import pstats

def profile_function(func, *args, **kwargs):
    profiler = cProfile.Profile()
    profiler.enable()
    result = func(*args, **kwargs)
    profiler.disable()
    
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)
    
    return result
```

### Memory Profiling

```python
# Monitor memory usage
import tracemalloc

tracemalloc.start()

# Your code here
result = orchestrator.run(task_graph, query)

current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")

tracemalloc.stop()
```

## 📞 Getting Help

### Before Asking for Help

1. **Check the logs** — Look for error messages and stack traces
2. **Verify configuration** — Ensure all settings are correct
3. **Test with minimal setup** — Try with basic configuration
4. **Check documentation** — Review relevant guides
5. **Search existing issues** — Check GitHub issues and discussions

### When Reporting Issues

Include the following information:

- **Error message** — Complete error text
- **Environment** — OS, Python version, arklex version
- **Configuration** — Relevant config files (remove sensitive data)
- **Steps to reproduce** — Clear reproduction steps
- **Expected vs actual behavior** — What you expected vs what happened
- **Logs** — Relevant log output

### Support Channels

- 📖 [Documentation](https://arklexai.github.io/Agent-First-Organization/)
- 💬 [GitHub Discussions](https://github.com/arklexai/Agent-First-Organization/discussions)
- 🐛 [Bug Reports](https://github.com/arklexai/Agent-First-Organization/issues)
- 📧 [Email Support](mailto:support@arklex.ai)
- 💬 [Discord Community](https://discord.gg/arklex)

---

For more detailed information on specific issues, see the [API Reference](API.md) and [Deployment Guide](DEPLOYMENT.md).
