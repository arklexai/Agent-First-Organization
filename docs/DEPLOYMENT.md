# 🚀 Production Deployment Guide

Complete guide for deploying Arklex AI agents to production with enterprise-grade reliability, security, and scalability.

## 📋 Table of Contents

- [Prerequisites](#-prerequisites)
- [Environment Setup](#-environment-setup)
- [Docker Deployment](#-docker-deployment)
- [Cloud Deployment](#-cloud-deployment)
- [Monitoring & Observability](#-monitoring--observability)
- [Security & Compliance](#-security--compliance)
- [Scaling & Performance](#-scaling--performance)
- [Backup & Recovery](#-backup--recovery)

## 🔧 Prerequisites

Before deploying to production, ensure you have:

- **Production-ready LLM API keys** with sufficient quotas
- **Vector database** (Milvus, Pinecone, or Weaviate)
- **SQL database** (MySQL, PostgreSQL, or SQLite)
- **Monitoring tools** (Prometheus, Grafana, or similar)
- **Load balancer** for high availability
- **SSL certificates** for HTTPS
- **Domain name** for your API

## ⚙️ Environment Setup

### Production Environment Variables

Create a production `.env` file:

```env
# =============================================================================
# REQUIRED: Production LLM Configuration
# =============================================================================

# OpenAI (recommended for production)
OPENAI_API_KEY=your_production_openai_key
OPENAI_ORG_ID=your_org_id

# Backup providers
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_gemini_key

# =============================================================================
# REQUIRED: Database Configuration
# =============================================================================

# Vector Database (Milvus)
MILVUS_URI=your_milvus_production_uri
MILVUS_USERNAME=your_milvus_username
MILVUS_PASSWORD=your_milvus_password

# SQL Database (MySQL)
MYSQL_USERNAME=your_mysql_username
MYSQL_PASSWORD=your_mysql_password
MYSQL_HOSTNAME=your_mysql_host
MYSQL_PORT=3306
MYSQL_DB_NAME=arklex_production
MYSQL_CONNECTION_TIMEOUT=10

# =============================================================================
# REQUIRED: Security Configuration
# =============================================================================

# JWT Authentication
JWT_SECRET=your_super_secure_jwt_secret_here
JWT_ALGORITHM=HS256
JWT_EXPIRATION=3600

# Rate Limiting
RATE_LIMIT_PER_MINUTE=100
RATE_LIMIT_BURST_SIZE=20

# API Security
API_KEY_HEADER=X-API-Key
API_KEYS=key1:user1,key2:user2

# =============================================================================
# REQUIRED: Production Settings
# =============================================================================

# Server Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_WORKER_CLASS=uvicorn.workers.UvicornWorker

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE=/var/log/arklex/app.log

# Monitoring
METRICS_ENABLED=true
HEALTH_CHECK_ENABLED=true
PROMETHEUS_ENABLED=true

# =============================================================================
# OPTIONAL: Enhanced Features
# =============================================================================

# Caching (Redis)
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=your_redis_password

# Web Search
TAVILY_API_KEY=your_tavily_key

# External Integrations
SHOPIFY_API_KEY=your_shopify_key
HUBSPOT_API_KEY=your_hubspot_key
GOOGLE_CALENDAR_CREDENTIALS=path/to/credentials.json

# =============================================================================
# OPTIONAL: Advanced Configuration
# =============================================================================

# Auto-scaling
AUTO_SCALE_ENABLED=true
MIN_WORKERS=2
MAX_WORKERS=10
SCALE_UP_THRESHOLD=0.8
SCALE_DOWN_THRESHOLD=0.2

# Circuit Breaker
CIRCUIT_BREAKER_ENABLED=true
CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
CIRCUIT_BREAKER_RECOVERY_TIMEOUT=60

# Backup
BACKUP_ENABLED=true
BACKUP_SCHEDULE=0 2 * * *  # Daily at 2 AM
BACKUP_RETENTION_DAYS=30
```

### Production Configuration File

Create `production_config.json`:

```json
{
  "name": "Production Agent",
  "description": "Production-ready agent with monitoring and security",
  "version": "1.0.0",
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
      "embedding_model": "text-embedding-ada-002",
      "top_k": 5,
      "similarity_threshold": 0.7,
      "cache_enabled": true,
      "cache_ttl": 3600
    },
    "database_worker": {
      "enabled": true,
      "connection_string": "mysql://user:pass@host:3306/arklex_production",
      "pool_size": 10,
      "max_overflow": 20,
      "echo": false,
      "ssl_mode": "require"
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
    "monitoring_middleware",
    "circuit_breaker_middleware"
  ],
  "security": {
    "authentication": "jwt",
    "rate_limiting": true,
    "input_validation": true,
    "ssl_required": true,
    "cors_enabled": true,
    "cors_origins": ["https://yourdomain.com"]
  },
  "monitoring": {
    "metrics_enabled": true,
    "health_checks": true,
    "logging": {
      "level": "INFO",
      "format": "json",
      "file": "/var/log/arklex/app.log"
    },
    "alerts": {
      "error_rate_threshold": 0.05,
      "latency_threshold": 5000,
      "memory_threshold": 0.9
    }
  }
}
```

## 🐳 Docker Deployment

### Dockerfile

```dockerfile
# Use Python 3.10 slim image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Create log directory
RUN mkdir -p /app/logs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the application
CMD ["python", "model_api.py", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  arklex:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - MILVUS_URI=${MILVUS_URI}
      - MYSQL_USERNAME=${MYSQL_USERNAME}
      - MYSQL_PASSWORD=${MYSQL_PASSWORD}
      - JWT_SECRET=${JWT_SECRET}
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    depends_on:
      - milvus
      - mysql
      - redis
    restart: unless-stopped
    networks:
      - arklex-network

  milvus:
    image: milvusdb/milvus:latest
    ports:
      - "19530:19530"
    environment:
      - MILVUS_HOST=milvus
      - MILVUS_PORT=19530
    volumes:
      - milvus_data:/var/lib/milvus
    restart: unless-stopped
    networks:
      - arklex-network

  mysql:
    image: mysql:8.0
    ports:
      - "3306:3306"
    environment:
      - MYSQL_ROOT_PASSWORD=${MYSQL_ROOT_PASSWORD}
      - MYSQL_DATABASE=arklex_production
      - MYSQL_USER=${MYSQL_USERNAME}
      - MYSQL_PASSWORD=${MYSQL_PASSWORD}
    volumes:
      - mysql_data:/var/lib/mysql
      - ./mysql/init:/docker-entrypoint-initdb.d
    restart: unless-stopped
    networks:
      - arklex-network

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    restart: unless-stopped
    networks:
      - arklex-network

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
    depends_on:
      - arklex
    restart: unless-stopped
    networks:
      - arklex-network

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    restart: unless-stopped
    networks:
      - arklex-network

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
    depends_on:
      - prometheus
    restart: unless-stopped
    networks:
      - arklex-network

volumes:
  milvus_data:
  mysql_data:
  redis_data:
  prometheus_data:
  grafana_data:

networks:
  arklex-network:
    driver: bridge
```

### Nginx Configuration

```nginx
# nginx/nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream arklex_backend {
        server arklex:8000;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

    server {
        listen 80;
        server_name yourdomain.com;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name yourdomain.com;

        # SSL configuration
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
        ssl_prefer_server_ciphers off;

        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";

        # Rate limiting
        limit_req zone=api burst=20 nodelay;

        # Proxy configuration
        location / {
            proxy_pass http://arklex_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_connect_timeout 30s;
            proxy_send_timeout 30s;
            proxy_read_timeout 30s;
        }

        # Health check endpoint
        location /health {
            proxy_pass http://arklex_backend/health;
            access_log off;
        }

        # Metrics endpoint
        location /metrics {
            proxy_pass http://arklex_backend/metrics;
            access_log off;
        }
    }
}
```

## ☁️ Cloud Deployment

### AWS Deployment

#### ECS with Fargate

```yaml
# task-definition.json
{
  "family": "arklex-agent",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/arklex-task-role",
  "containerDefinitions": [
    {
      "name": "arklex",
      "image": "your-registry/arklex:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "OPENAI_API_KEY",
          "value": "your-openai-key"
        }
      ],
      "secrets": [
        {
          "name": "JWT_SECRET",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:arklex-jwt-secret"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/arklex",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

#### Application Load Balancer

```yaml
# alb.yaml
Resources:
  ApplicationLoadBalancer:
    Type: AWS::ElasticLoadBalancingV2::LoadBalancer
    Properties:
      Name: arklex-alb
      Scheme: internet-facing
      Type: application
      SecurityGroups:
        - !Ref ALBSecurityGroup
      Subnets:
        - !Ref PublicSubnet1
        - !Ref PublicSubnet2

  ALBTargetGroup:
    Type: AWS::ElasticLoadBalancingV2::TargetGroup
    Properties:
      Name: arklex-tg
      Port: 8000
      Protocol: HTTP
      VpcId: !Ref VPC
      TargetType: ip
      HealthCheckPath: /health
      HealthCheckIntervalSeconds: 30
      HealthyThresholdCount: 2
      UnhealthyThresholdCount: 3

  ALBListener:
    Type: AWS::ElasticLoadBalancingV2::Listener
    Properties:
      LoadBalancerArn: !Ref ApplicationLoadBalancer
      Port: 443
      Protocol: HTTPS
      Certificates:
        - CertificateArn: !Ref SSLCertificate
      DefaultActions:
        - Type: forward
          TargetGroupArn: !Ref ALBTargetGroup
```

### Google Cloud Platform

#### Cloud Run

```yaml
# cloud-run.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: arklex-agent
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "2"
        autoscaling.knative.dev/maxScale: "10"
    spec:
      containerConcurrency: 100
      timeoutSeconds: 300
      containers:
      - image: gcr.io/your-project/arklex:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: arklex-secrets
              key: openai-api-key
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: arklex-secrets
              key: jwt-secret
        resources:
          limits:
            cpu: "1000m"
            memory: "2Gi"
          requests:
            cpu: "500m"
            memory: "1Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

## 📊 Monitoring & Observability

### Prometheus Configuration

```yaml
# prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "arklex_rules.yml"

scrape_configs:
  - job_name: 'arklex'
    static_configs:
      - targets: ['arklex:8000']
    metrics_path: /metrics
    scrape_interval: 10s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Grafana Dashboards

Create custom dashboards for:

- **Response Time** — Average, 95th percentile, 99th percentile
- **Throughput** — Requests per second, successful vs failed
- **Error Rate** — Error percentage, error types
- **Resource Usage** — CPU, memory, disk I/O
- **LLM Metrics** — Token usage, API costs, model performance
- **Database Metrics** — Connection pool, query performance
- **Custom Metrics** — Business-specific KPIs

### Logging Configuration

```python
# logging_config.py
import logging
import logging.handlers
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        
        return json.dumps(log_entry)

def setup_logging():
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        '/var/log/arklex/app.log',
        maxBytes=100*1024*1024,  # 100MB
        backupCount=5
    )
    file_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(file_handler)
    
    # Console handler for development
    if os.getenv('ENVIRONMENT') == 'development':
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(console_handler)
```

## 🔐 Security & Compliance

### Authentication & Authorization

```python
# auth_middleware.py
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from datetime import datetime, timedelta

security = HTTPBearer()

class AuthMiddleware:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
    
    def create_token(self, user_id: str, expires_delta: timedelta = None):
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=1)
        
        to_encode = {"sub": user_id, "exp": expire}
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm="HS256")
        return encoded_jwt
    
    def verify_token(self, credentials: HTTPAuthorizationCredentials = Depends(security)):
        try:
            payload = jwt.decode(credentials.credentials, self.secret_key, algorithms=["HS256"])
            user_id = payload.get("sub")
            if user_id is None:
                raise HTTPException(status_code=401, detail="Invalid token")
            return user_id
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")
```

### Rate Limiting

```python
# rate_limit_middleware.py
from fastapi import HTTPException
import time
from collections import defaultdict
import threading

class RateLimitMiddleware:
    def __init__(self, requests_per_minute: int = 100, burst_size: int = 20):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.requests = defaultdict(list)
        self.lock = threading.Lock()
    
    def check_rate_limit(self, client_id: str):
        with self.lock:
            now = time.time()
            client_requests = self.requests[client_id]
            
            # Remove old requests
            client_requests[:] = [req_time for req_time in client_requests 
                                if now - req_time < 60]
            
            # Check rate limit
            if len(client_requests) >= self.requests_per_minute:
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            
            # Add current request
            client_requests.append(now)
```

### Input Validation

```python
# validation_middleware.py
from pydantic import BaseModel, validator
from typing import Optional
import re

class ChatRequest(BaseModel):
    query: str
    context: Optional[dict] = None
    user_id: Optional[str] = None
    
    @validator('query')
    def validate_query(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Query cannot be empty')
        if len(v) > 10000:
            raise ValueError('Query too long (max 10000 characters)')
        
        # Check for potentially malicious content
        suspicious_patterns = [
            r'<script.*?>',
            r'javascript:',
            r'data:text/html',
            r'vbscript:'
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError('Query contains potentially malicious content')
        
        return v.strip()
    
    @validator('context')
    def validate_context(cls, v):
        if v and not isinstance(v, dict):
            raise ValueError('Context must be a dictionary')
        return v
```

## ⚡ Scaling & Performance

### Auto-scaling Configuration

```python
# auto_scaling.py
import asyncio
import psutil
from typing import List
import time

class AutoScaler:
    def __init__(self, min_workers: int = 2, max_workers: int = 10):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.current_workers = min_workers
        self.scale_up_threshold = 0.8
        self.scale_down_threshold = 0.2
        self.cooldown_period = 300  # 5 minutes
        self.last_scale_time = 0
    
    async def monitor_and_scale(self):
        while True:
            try:
                # Get current metrics
                cpu_usage = psutil.cpu_percent(interval=1)
                memory_usage = psutil.virtual_memory().percent
                load_average = psutil.getloadavg()[0]
                
                # Calculate load score
                load_score = (cpu_usage + memory_usage) / 200
                
                # Check if scaling is needed
                if time.time() - self.last_scale_time > self.cooldown_period:
                    if load_score > self.scale_up_threshold and self.current_workers < self.max_workers:
                        await self.scale_up()
                    elif load_score < self.scale_down_threshold and self.current_workers > self.min_workers:
                        await self.scale_down()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                print(f"Auto-scaling error: {e}")
                await asyncio.sleep(60)
    
    async def scale_up(self):
        self.current_workers = min(self.current_workers + 1, self.max_workers)
        self.last_scale_time = time.time()
        print(f"Scaled up to {self.current_workers} workers")
    
    async def scale_down(self):
        self.current_workers = max(self.current_workers - 1, self.min_workers)
        self.last_scale_time = time.time()
        print(f"Scaled down to {self.current_workers} workers")
```

### Caching Strategy

```python
# caching.py
import redis
import json
import hashlib
from typing import Any, Optional

class CacheManager:
    def __init__(self, redis_url: str, default_ttl: int = 3600):
        self.redis = redis.from_url(redis_url)
        self.default_ttl = default_ttl
    
    def _generate_key(self, prefix: str, data: Any) -> str:
        """Generate cache key from data"""
        data_str = json.dumps(data, sort_keys=True)
        hash_value = hashlib.md5(data_str.encode()).hexdigest()
        return f"{prefix}:{hash_value}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            value = self.redis.get(key)
            return json.loads(value) if value else None
        except Exception as e:
            print(f"Cache get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        try:
            ttl = ttl or self.default_ttl
            return self.redis.setex(key, ttl, json.dumps(value))
        except Exception as e:
            print(f"Cache set error: {e}")
            return False
    
    async def cache_rag_results(self, query: str, documents: list, ttl: int = 3600):
        """Cache RAG search results"""
        key = self._generate_key("rag", query)
        await self.set(key, documents, ttl)
    
    async def get_cached_rag_results(self, query: str) -> Optional[list]:
        """Get cached RAG search results"""
        key = self._generate_key("rag", query)
        return await self.get(key)
```

## 💾 Backup & Recovery

### Database Backup

```bash
#!/bin/bash
# backup.sh

# Configuration
BACKUP_DIR="/backups"
MYSQL_HOST="localhost"
MYSQL_USER="backup_user"
MYSQL_PASSWORD="backup_password"
MYSQL_DATABASE="arklex_production"
RETENTION_DAYS=30

# Create backup directory
mkdir -p $BACKUP_DIR

# Generate backup filename
BACKUP_FILE="$BACKUP_DIR/arklex_$(date +%Y%m%d_%H%M%S).sql"

# Create MySQL backup
mysqldump -h $MYSQL_HOST -u $MYSQL_USER -p$MYSQL_PASSWORD \
    --single-transaction \
    --routines \
    --triggers \
    $MYSQL_DATABASE > $BACKUP_FILE

# Compress backup
gzip $BACKUP_FILE

# Remove old backups
find $BACKUP_DIR -name "arklex_*.sql.gz" -mtime +$RETENTION_DAYS -delete

echo "Backup completed: $BACKUP_FILE.gz"
```

### Vector Database Backup

```python
# vector_backup.py
import os
import json
from datetime import datetime
from pymilvus import connections, Collection

class VectorBackup:
    def __init__(self, milvus_uri: str, backup_dir: str):
        self.milvus_uri = milvus_uri
        self.backup_dir = backup_dir
        connections.connect("default", uri=milvus_uri)
    
    def backup_collection(self, collection_name: str):
        """Backup a Milvus collection"""
        try:
            collection = Collection(collection_name)
            collection.load()
            
            # Get collection schema
            schema = collection.schema
            
            # Export data
            results = collection.query(
                expr="",
                output_fields=schema.fields,
                limit=collection.num_entities
            )
            
            # Create backup file
            backup_file = os.path.join(
                self.backup_dir,
                f"{collection_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            with open(backup_file, 'w') as f:
                json.dump({
                    'schema': schema.to_dict(),
                    'data': results,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
            
            print(f"Backup completed: {backup_file}")
            return backup_file
            
        except Exception as e:
            print(f"Backup failed: {e}")
            return None
    
    def restore_collection(self, backup_file: str, collection_name: str):
        """Restore a Milvus collection from backup"""
        try:
            with open(backup_file, 'r') as f:
                backup_data = json.load(f)
            
            # Create collection from schema
            schema = backup_data['schema']
            collection = Collection(collection_name, schema)
            
            # Insert data
            if backup_data['data']:
                collection.insert(backup_data['data'])
            
            print(f"Restore completed: {collection_name}")
            return True
            
        except Exception as e:
            print(f"Restore failed: {e}")
            return False
```

### Disaster Recovery Plan

1. **Regular Backups**
   - Daily database backups
   - Weekly vector database backups
   - Configuration file backups

2. **Recovery Procedures**
   - Database restoration scripts
   - Vector database restoration
   - Configuration restoration

3. **Testing**
   - Monthly recovery drills
   - Backup integrity checks
   - Performance testing after recovery

---

For more detailed information on specific deployment scenarios, see the [Architecture Guide](ARCHITECTURE.md) and [Troubleshooting Guide](TROUBLESHOOTING.md).
