# Security Improvements

## 1. Authentication & Authorization

### Current Issues

- **Weak API Key Management**: API keys stored in environment variables without proper rotation
- **Missing JWT Implementation**: No proper JWT token-based authentication
- **No Role-Based Access Control**: No RBAC system for different user types
- **Insecure Default Configurations**: Default configurations allow wide access

### Proposed Solutions

#### JWT Authentication Implementation

```python
# arklex/core/auth/jwt_handler.py
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

class JWTHandler:
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
    
    def create_token(self, user_id: str, roles: list[str], expires_delta: timedelta = None) -> str:
        """Create JWT token with user information."""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=1)
        
        to_encode = {
            "sub": user_id,
            "roles": roles,
            "exp": expire,
            "iat": datetime.utcnow()
        }
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")
```

#### Role-Based Access Control

```python
# arklex/core/auth/rbac.py
from enum import Enum
from typing import Set, Dict, Any
from functools import wraps

class Permission(Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"

class Role(Enum):
    USER = "user"
    ADMIN = "admin"
    SYSTEM = "system"

class RBACManager:
    def __init__(self):
        self.role_permissions = {
            Role.USER: {Permission.READ},
            Role.ADMIN: {Permission.READ, Permission.WRITE, Permission.DELETE},
            Role.SYSTEM: {Permission.READ, Permission.WRITE, Permission.DELETE, Permission.ADMIN}
        }
    
    def has_permission(self, user_roles: Set[Role], required_permission: Permission) -> bool:
        """Check if user has required permission."""
        for role in user_roles:
            if required_permission in self.role_permissions.get(role, set()):
                return True
        return False
    
    def require_permission(self, permission: Permission):
        """Decorator to require specific permission."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Extract user roles from request context
                user_roles = get_user_roles_from_context()
                if not self.has_permission(user_roles, permission):
                    raise HTTPException(status_code=403, detail="Insufficient permissions")
                return func(*args, **kwargs)
            return wrapper
        return decorator
```

## 2. Input Validation & Sanitization

### Current Issues

- **Missing Input Validation**: No comprehensive input validation system
- **SQL Injection Vulnerabilities**: Direct database queries without proper sanitization
- **XSS Vulnerabilities**: No HTML sanitization for user inputs
- **File Upload Security**: No validation for uploaded files

### Proposed Solutions

#### Input Validation Framework

```python
# arklex/core/validation/validators.py
from pydantic import BaseModel, validator, Field
from typing import Any, Optional
import re
import html

class InputValidator:
    @staticmethod
    def sanitize_html(text: str) -> str:
        """Sanitize HTML content to prevent XSS."""
        return html.escape(text)
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_file_upload(filename: str, allowed_extensions: set) -> bool:
        """Validate file upload security."""
        if '.' not in filename:
            return False
        ext = filename.rsplit('.', 1)[1].lower()
        return ext in allowed_extensions

class SecureRequestModel(BaseModel):
    """Base model for secure request validation."""
    class Config:
        extra = "forbid"  # Reject unknown fields
    
    @validator('*', pre=True)
    def sanitize_strings(cls, v):
        if isinstance(v, str):
            return InputValidator.sanitize_html(v)
        return v
```

#### SQL Injection Prevention

```python
# arklex/core/database/secure_queries.py
from typing import Any, Dict, List
import mysql.connector
from mysql.connector import pooling

class SecureDatabaseManager:
    def __init__(self, connection_pool):
        self.pool = connection_pool
    
    def execute_secure_query(self, query: str, params: tuple = None) -> List[Dict[str, Any]]:
        """Execute query with parameterized statements to prevent SQL injection."""
        connection = self.pool.get_connection()
        try:
            cursor = connection.cursor(dictionary=True)
            cursor.execute(query, params)
            result = cursor.fetchall()
            return result
        finally:
            cursor.close()
            connection.close()
    
    def execute_secure_insert(self, table: str, data: Dict[str, Any]) -> int:
        """Secure insert with parameterized queries."""
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['%s'] * len(data))
        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        
        connection = self.pool.get_connection()
        try:
            cursor = connection.cursor()
            cursor.execute(query, tuple(data.values()))
            connection.commit()
            return cursor.lastrowid
        finally:
            cursor.close()
            connection.close()
```

## 3. API Security

### Current Issues

- **Missing Rate Limiting**: No rate limiting implementation
- **No Request Validation**: Incoming requests not properly validated
- **CORS Misconfiguration**: Overly permissive CORS settings
- **No API Versioning**: No versioning strategy for API changes

### Proposed Solutions

#### Rate Limiting Implementation

```python
# arklex/core/security/rate_limiter.py
import time
from collections import defaultdict
from typing import Dict, Tuple
import redis

class RateLimiter:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.default_limit = 100  # requests per minute
        self.default_window = 60  # seconds
    
    def is_allowed(self, client_id: str, limit: int = None, window: int = None) -> bool:
        """Check if request is allowed based on rate limits."""
        limit = limit or self.default_limit
        window = window or self.default_window
        
        current_time = int(time.time())
        window_start = current_time - window
        
        # Use Redis sorted set for efficient rate limiting
        key = f"rate_limit:{client_id}"
        
        # Remove old entries
        self.redis.zremrangebyscore(key, 0, window_start)
        
        # Count current requests
        current_count = self.redis.zcard(key)
        
        if current_count >= limit:
            return False
        
        # Add current request
        self.redis.zadd(key, {str(current_time): current_time})
        self.redis.expire(key, window)
        
        return True
    
    def get_remaining_requests(self, client_id: str) -> int:
        """Get remaining requests for client."""
        key = f"rate_limit:{client_id}"
        current_count = self.redis.zcard(key)
        return max(0, self.default_limit - current_count)
```

#### Secure API Middleware

```python
# arklex/core/middleware/security_middleware.py
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
import time
from typing import Callable

class SecurityMiddleware:
    def __init__(self, rate_limiter, jwt_handler):
        self.rate_limiter = rate_limiter
        self.jwt_handler = jwt_handler
    
    async def __call__(self, request: Request, call_next: Callable):
        # Rate limiting
        client_id = self.get_client_id(request)
        if not self.rate_limiter.is_allowed(client_id):
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded"}
            )
        
        # Request size validation
        if request.headers.get("content-length"):
            content_length = int(request.headers["content-length"])
            if content_length > 10 * 1024 * 1024:  # 10MB limit
                return JSONResponse(
                    status_code=413,
                    content={"error": "Request too large"}
                )
        
        # Security headers
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        return response
    
    def get_client_id(self, request: Request) -> str:
        """Extract client identifier for rate limiting."""
        # Use API key or IP address
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"api_key:{api_key}"
        return f"ip:{request.client.host}"
```

## 4. Data Protection

### Current Issues

- **Sensitive Data Exposure**: API keys and secrets in logs
- **No Data Encryption**: Sensitive data not encrypted at rest
- **Missing Audit Logging**: No comprehensive audit trail
- **Insecure Configuration**: Configuration files with sensitive data

### Proposed Solutions

#### Sensitive Data Protection

```python
# arklex/core/security/data_protection.py
import os
from cryptography.fernet import Fernet
from typing import Any, Dict
import json

class DataProtector:
    def __init__(self, encryption_key: bytes = None):
        self.key = encryption_key or Fernet.generate_key()
        self.cipher = Fernet(self.key)
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        return self.cipher.decrypt(encrypted_data.encode()).decode()
    
    def mask_sensitive_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mask sensitive fields in data structures."""
        sensitive_fields = {'password', 'api_key', 'token', 'secret'}
        masked_data = data.copy()
        
        for key, value in masked_data.items():
            if key.lower() in sensitive_fields and isinstance(value, str):
                masked_data[key] = '*' * len(value)
        
        return masked_data

class SecureLogger:
    def __init__(self, data_protector: DataProtector):
        self.data_protector = data_protector
    
    def log_securely(self, message: str, data: Dict[str, Any] = None):
        """Log message with sensitive data masked."""
        if data:
            masked_data = self.data_protector.mask_sensitive_fields(data)
            message = f"{message} - Data: {masked_data}"
        
        # Use structured logging
        log_entry = {
            "timestamp": time.time(),
            "message": message,
            "level": "INFO"
        }
        print(json.dumps(log_entry))
```

#### Audit Logging

```python
# arklex/core/audit/audit_logger.py
import json
import time
from typing import Dict, Any, Optional
from datetime import datetime

class AuditLogger:
    def __init__(self, log_file: str = "audit.log"):
        self.log_file = log_file
    
    def log_action(self, user_id: str, action: str, resource: str, 
                   details: Dict[str, Any] = None, success: bool = True):
        """Log user action for audit purposes."""
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "action": action,
            "resource": resource,
            "success": success,
            "details": details or {},
            "ip_address": self.get_client_ip(),
            "user_agent": self.get_user_agent()
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(audit_entry) + '\n')
    
    def get_client_ip(self) -> str:
        """Get client IP address from request context."""
        # Implementation depends on your request context
        return "unknown"
    
    def get_user_agent(self) -> str:
        """Get user agent from request context."""
        # Implementation depends on your request context
        return "unknown"
```

## 5. Configuration Security

### Current Issues

- **Hardcoded Secrets**: Secrets in configuration files
- **Insecure Defaults**: Default configurations with security vulnerabilities
- **No Secret Rotation**: No mechanism for rotating secrets
- **Environment Variable Exposure**: Sensitive data in environment variables

### Proposed Solutions

#### Secure Configuration Management

```python
# arklex/core/config/secure_config.py
import os
from typing import Dict, Any, Optional
from pydantic import BaseSettings, SecretStr

class SecureSettings(BaseSettings):
    """Secure configuration settings with encrypted secrets."""
    
    # Database
    database_url: SecretStr
    database_pool_size: int = 10
    
    # API Keys (encrypted)
    openai_api_key: SecretStr
    anthropic_api_key: Optional[SecretStr] = None
    
    # Security
    jwt_secret: SecretStr
    encryption_key: SecretStr
    
    # Rate Limiting
    rate_limit_per_minute: int = 100
    rate_limit_burst_size: int = 20
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

class SecretManager:
    def __init__(self, encryption_key: str):
        self.encryption_key = encryption_key
    
    def get_secret(self, secret_name: str) -> str:
        """Retrieve and decrypt secret."""
        # Implementation for secure secret retrieval
        # Could integrate with AWS Secrets Manager, HashiCorp Vault, etc.
        pass
    
    def rotate_secret(self, secret_name: str) -> str:
        """Rotate secret and return new value."""
        # Implementation for secret rotation
        pass
```

## 6. Security Testing

### Current Issues

- **No Security Testing**: No automated security testing
- **Missing Vulnerability Scanning**: No dependency vulnerability scanning
- **No Penetration Testing**: No penetration testing framework
- **Insufficient Security Monitoring**: No security event monitoring

### Proposed Solutions

#### Security Testing Framework

```python
# tests/security/test_security.py
import pytest
from arklex.core.auth.jwt_handler import JWTHandler
from arklex.core.security.rate_limiter import RateLimiter
from arklex.core.validation.validators import InputValidator

class TestSecurityFeatures:
    def test_jwt_token_creation_and_verification(self):
        """Test JWT token creation and verification."""
        handler = JWTHandler("test_secret")
        token = handler.create_token("user123", ["user"])
        payload = handler.verify_token(token)
        assert payload["sub"] == "user123"
    
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        # Mock Redis client
        limiter = RateLimiter(mock_redis_client)
        
        # Test within limits
        assert limiter.is_allowed("client1") == True
        
        # Test exceeding limits
        for _ in range(101):
            limiter.is_allowed("client2")
        assert limiter.is_allowed("client2") == False
    
    def test_input_sanitization(self):
        """Test input sanitization."""
        validator = InputValidator()
        
        # Test XSS prevention
        malicious_input = "<script>alert('xss')</script>"
        sanitized = validator.sanitize_html(malicious_input)
        assert "<script>" not in sanitized
    
    def test_sql_injection_prevention(self):
        """Test SQL injection prevention."""
        # Test with malicious input
        malicious_input = "'; DROP TABLE users; --"
        # Should be properly escaped by parameterized queries
        pass
```

#### Security Scanning Configuration

```yaml
# .github/workflows/security-scan.yml
name: Security Scan
on: [push, pull_request]

jobs:
  security-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Run Bandit (Python security linter)
        run: |
          pip install bandit
          bandit -r arklex/ -f json -o bandit-report.json
      
      - name: Run Safety (dependency vulnerability scanner)
        run: |
          pip install safety
          safety check --json --output safety-report.json
      
      - name: Run Semgrep (static analysis)
        run: |
          pip install semgrep
          semgrep --config=auto arklex/ --json --output semgrep-report.json
      
      - name: Upload security reports
        uses: actions/upload-artifact@v3
        with:
          name: security-reports
          path: |
            bandit-report.json
            safety-report.json
            semgrep-report.json
```

## 7. Implementation Priority

### High Priority (Immediate - 1-2 weeks)

1. **JWT Authentication Implementation**
2. **Rate Limiting**
3. **Input Validation Framework**
4. **Security Headers**

### Medium Priority (1-2 months)

1. **Role-Based Access Control**
2. **Audit Logging**
3. **Data Encryption**
4. **Security Testing Framework**

### Low Priority (3-6 months)

1. **Advanced Security Monitoring**
2. **Penetration Testing**
3. **Secret Management Integration**
4. **Security Documentation**

## 8. Success Metrics

- [ ] Zero critical security vulnerabilities
- [ ] 100% API endpoints protected with authentication
- [ ] All sensitive data encrypted at rest
- [ ] Rate limiting implemented for all endpoints
- [ ] Security testing automated in CI/CD
- [ ] Audit logging for all sensitive operations
- [ ] Security headers implemented on all responses
- [ ] Input validation on all user inputs
