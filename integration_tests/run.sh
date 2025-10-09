#!/bin/bash

# Exit on error
set -e

# Accept test type parameter (mock or live)
TEST_TYPE=${1:-"mock"}

# Validate test type
if [[ "$TEST_TYPE" != "mock" && "$TEST_TYPE" != "live" ]]; then
    echo "Error: Invalid test type. Must be 'mock' or 'live'"
    exit 1
fi

echo "Running $TEST_TYPE integration tests"

# Start Milvus
echo "Starting Milvus..."
bash integration_tests/utils/vector_database/standalone_milvus.sh start
python integration_tests/utils/vector_database/init_milvus_db.py

# Cleanup function to ensure Milvus is stopped and deleted
cleanup() {
    echo "Stopping and deleting Milvus..."
    bash integration_tests/utils/vector_database/standalone_milvus.sh stop || true
    bash integration_tests/utils/vector_database/standalone_milvus.sh delete || true
}

# Set trap to run cleanup on exit (whether success or failure)
trap cleanup EXIT

# Integration tests
echo "Test type: ${TEST_TYPE}"
## Mock tests
if [[ "$TEST_TYPE" == "mock" ]]; then
    pytest integration_tests/mock_tests -v
fi
## Live tests
if [[ "$TEST_TYPE" == "live" ]]; then
    pytest integration_tests/live_tests -v
fi
