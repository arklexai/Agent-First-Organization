#!/bin/bash

# Integration Tests Runner Script
# This script runs all integration tests with the correct environment variables

set -e

echo "Running integration tests with ARKLEX_TEST_ENV=local and KMP_DUPLICATE_LIB_OK=TRUE..."

# Set environment variables
export ARKLEX_TEST_ENV=local
export KMP_DUPLICATE_LIB_OK=TRUE

# Run all integration tests from the integration directory
cd tests/integration
pytest . -v

echo "✅ All integration tests completed successfully!"
