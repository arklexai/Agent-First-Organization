# Integration Tests

This document explains how to run the integration tests in the `tests/integration/` directory and the solution to common issues.

## Quick Start

To run all integration tests, use the provided script:

```bash
# From the project root
./tests/integration/run_integration_tests.sh

# Or from within the integration tests directory
cd tests/integration
./run_integration_tests.sh
```

This script automatically sets the required environment variables and runs all integration tests.

## Manual Execution

If you prefer to run tests manually, use these environment variables:

```bash
KMP_DUPLICATE_LIB_OK=TRUE ARKLEX_TEST_ENV=local pytest tests/integration/ -v
```

## Test Files

The integration tests include:

- **HITL Server Tests** (`test_hitl_server.py`): 11 tests covering Human-in-the-Loop functionality
- **Shopify Tests** (`test_shopify.py`): 26 tests covering Shopify integration tools
- **HubSpot Tests** (`test_hubspot.py`): 17 tests covering HubSpot integration tools
- **Milvus Filter Tests** (`test_milvus_filter.py`): 12 tests covering Milvus vector search functionality

## Environment Variables

### Required Variables

- `ARKLEX_TEST_ENV=local`: Sets the test environment to local mode
- `KMP_DUPLICATE_LIB_OK=TRUE`: **Critical for macOS** - Prevents FAISS/OpenMP crashes

### Optional Variables

- `OPENAI_API_KEY`: Test API key (defaults to "test_key")
- `MYSQL_USERNAME`: Test MySQL username (defaults to "test_user")
- `MYSQL_PASSWORD`: Test MySQL password (defaults to "test_password")
- `MYSQL_HOSTNAME`: Test MySQL hostname (defaults to "localhost")
- `MYSQL_PORT`: Test MySQL port (defaults to "3306")
- `MYSQL_DB_NAME`: Test MySQL database name (defaults to "test_db")
- `DATA_DIR`: Test data directory (defaults to "./examples/hitl_server")
- `TESTING`: Testing mode flag (defaults to "true")
- `LOG_LEVEL`: Log level for tests (defaults to "WARNING")

## Common Issues and Solutions

### FAISS/OpenMP Crash on macOS

**Problem**: Tests crash with error:

```
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
Fatal Python error: Aborted
```

**Solution**: Set the environment variable `KMP_DUPLICATE_LIB_OK=TRUE` before running tests:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
pytest tests/integration/ -v
```

This is a known issue with FAISS and OpenMP on macOS where multiple copies of the OpenMP runtime are linked into the program.

### Running Specific Test Files

```bash
# Run only HITL tests
KMP_DUPLICATE_LIB_OK=TRUE ARKLEX_TEST_ENV=local pytest tests/integration/test_hitl_server.py -v

# Run only Shopify tests
KMP_DUPLICATE_LIB_OK=TRUE ARKLEX_TEST_ENV=local pytest tests/integration/test_shopify.py -v

# Run only HubSpot tests
KMP_DUPLICATE_LIB_OK=TRUE ARKLEX_TEST_ENV=local pytest tests/integration/test_hubspot.py -v

# Run only Milvus tests
KMP_DUPLICATE_LIB_OK=TRUE ARKLEX_TEST_ENV=local pytest tests/integration/test_milvus_filter.py -v
```

### Running Specific Tests

```bash
# Run a specific test
KMP_DUPLICATE_LIB_OK=TRUE ARKLEX_TEST_ENV=local pytest tests/integration/test_hitl_server.py::TestHITLServerIntegration::test_hitl_chat_flag_activation -v

# Run tests with specific markers
KMP_DUPLICATE_LIB_OK=TRUE ARKLEX_TEST_ENV=local pytest tests/integration/ -m "hitl" -v
```

## Test Coverage

The integration tests provide comprehensive coverage of:

- **HITL Functionality**: Chat flag activation, MC flag activation, conversation flows
- **RAG Operations**: Product question responses, vector search functionality
- **External Integrations**: Shopify API tools, HubSpot API tools
- **Error Handling**: API exceptions, authentication errors, edge cases
- **State Management**: Conversation history, node transitions, worker configurations

## Test Results

All 66 integration tests should pass when run with the correct environment variables:

- ✅ 11 HITL Server tests
- ✅ 26 Shopify tests  
- ✅ 17 HubSpot tests
- ✅ 12 Milvus Filter tests

## Alternative Test Runner

You can also use the HITL test runner script:

```bash
KMP_DUPLICATE_LIB_OK=TRUE ARKLEX_TEST_ENV=local python tests/integration/run_hitl_tests.py -v
```

## Troubleshooting

1. **Import Errors**: Ensure the project root is in the Python path
2. **Environment Variables**: Check that all required environment variables are set
3. **Mock Issues**: Verify that external services are properly mocked
4. **Test Data**: Ensure test data files exist in the expected locations
5. **FAISS Crashes**: Always use `KMP_DUPLICATE_LIB_OK=TRUE` on macOS

## Debug Mode

Run tests with verbose output and full tracebacks:

```bash
KMP_DUPLICATE_LIB_OK=TRUE ARKLEX_TEST_ENV=local pytest tests/integration/ -v -s --tb=long
```
