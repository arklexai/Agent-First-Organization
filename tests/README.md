# Test Suite Documentation

## Overview

This test suite validates the behavior of different orchestrator types in the Arklex framework. It supports both local testing with mocked LLM responses and integration testing with real LLM responses.

## Running Tests

### Local Testing (with Mocked LLM)

To run tests locally with mocked LLM responses:

```bash
export ARKLEX_TEST_ENV=local
python3 -m pytest tests/test_resources.py -v
```

This will use predefined responses for LLM interactions, making tests:

- Faster (no API calls)
- More predictable
- Independent of external services

### Integration Testing (with Real LLM)

To run tests with real LLM responses (default):

```bash
python3 -m pytest tests/test_resources.py -v
```

This will make actual API calls to the LLM service, useful for:

- Validating real-world behavior
- Testing API integration
- Catching service-specific issues

### Running Specific Test Cases

To run a specific test case:

```bash
# Run a specific orchestrator test
python3 -m pytest tests/test_resources.py::test_resources[MCWorkerOrchestrator-mc_worker_taskgraph.json-mc_worker_testcases.json] -v

# Run tests matching a pattern
python3 -m pytest tests/test_resources.py -k "MCWorker" -v

# Run tests with specific markers
python3 -m pytest tests/test_resources.py -m "not slow" -v
```

## Test Structure

The test suite is organized as follows:

```
tests/
├── README.md                 # This documentation
├── test_resources.py        # Main test runner
├── data/                    # Test data directory
│   ├── mc_worker_taskgraph.json
│   ├── mc_worker_testcases.json
│   ├── message_worker_taskgraph.json
│   ├── message_worker_testcases.json
│   ├── shopify_tool_taskgraph.json
│   └── shopify_tool_testcases.json
└── utils/                   # Test utilities
    ├── utils.py            # Core test utilities
    ├── utils_workers.py    # Worker-specific test utilities
    └── utils_tools.py      # Tool-specific test utilities
```

## Adding New Tests

### 1. Create Test Configuration

Create a new taskgraph configuration file in `tests/data/`:

```json
{
    "nodes": [
        ["0", {"type": "start", "attribute": {"value": "Hello! How can I help you?"}}],
        ["1", {"type": "message", "attribute": {"value": "What would you like to know?"}}]
    ],
    "edges": [
        ["0", "1"]
    ]
}
```

### 2. Create Test Cases

Create a test cases file in `tests/data/`:

```json
[
    {
        "user_utterance": ["Hello", "What products do you have?"],
        "expected_taskgraph_path": ["0", "1"]
    }
]
```

### 3. Add Test Case to Runner

Add your test case to `TEST_CASES` in `tests/test_resources.py`:

```python
TEST_CASES: List[TestCaseConfig] = [
    # ... existing test cases ...
    TestCaseConfig(
        YourNewOrchestrator,
        "your_taskgraph.json",
        "your_testcases.json"
    ),
]
```

## Example Test Cases

### Basic Conversation Flow

```json
{
    "user_utterance": [
        "Hello",
        "I want to buy a product",
        "Tell me about Product A"
    ],
    "expected_taskgraph_path": ["0", "1", "2"]
}
```

### Error Handling

```json
{
    "user_utterance": [
        "Hello",
        "Invalid input",
        "What products do you have?"
    ],
    "expected_taskgraph_path": ["0", "1", "2"]
}
```

### Complex Multi-turn Dialog

```json
{
    "user_utterance": [
        "Hello",
        "I want to buy a product",
        "Tell me about Product A",
        "What's the price?",
        "I'll take it"
    ],
    "expected_taskgraph_path": ["0", "1", "2", "3", "4"]
}
```

### Conditional Branching

```json
{
    "user_utterance": [
        "Hello",
        "I want to buy a product",
        "Tell me about Product A",
        "What's the price?",
        "That's too expensive",
        "What about Product B?"
    ],
    "expected_taskgraph_path": ["0", "1", "2", "3", "4", "5"]
}
```

### Slot Filling

```json
{
    "user_utterance": [
        "Hello",
        "I want to order a product",
        "Product A",
        "Size large",
        "Color blue",
        "Yes, that's correct"
    ],
    "expected_taskgraph_path": ["0", "1", "2", "3", "4", "5"],
    "expected_slots": {
        "product": "Product A",
        "size": "large",
        "color": "blue"
    }
}
```

## Mock LLM Responses

When running tests locally, the mock LLM provides predefined responses based on the conversation context. Here's how to customize these responses in `tests/utils/utils.py`:

```python
def dummy_invoke(*args, **kwargs):
    user_msg = get_last_user_message(args, kwargs)
    if user_msg == "What products do you have?":
        return DummyAIMessage(
            '{"name": "respond", "arguments": {"content": "We have products A, B, and C", "node_id": "1"}}'
        )
    # Add more response patterns as needed
```

### Common Response Patterns

1. **Product Information**

```python
if "product" in user_msg.lower():
    return DummyAIMessage(
        '{"name": "respond", "arguments": {"content": "This product features...", "node_id": "2"}}'
    )
```

2. **Price Queries**

```python
if "price" in user_msg.lower():
    return DummyAIMessage(
        '{"name": "respond", "arguments": {"content": "The price is $99.99", "node_id": "3"}}'
    )
```

3. **Confirmation Handling**

```python
if any(word in user_msg.lower() for word in ["yes", "correct", "right"]):
    return DummyAIMessage(
        '{"name": "respond", "arguments": {"content": "Great! Let's proceed.", "node_id": "4"}}'
    )
```

## Known Warnings

The test suite may show several warnings that are safe to ignore:

### NumPy Deprecation Warning

```
DeprecationWarning: numpy.core._multiarray_umath is deprecated and has been renamed to numpy._core._multiarray_umath
```

This warning comes from the FAISS library's internal usage of NumPy. It's a deprecation warning about NumPy's internal API usage and doesn't affect functionality. This will be fixed in future versions of FAISS.

### SWIG Type Warnings

```
DeprecationWarning: builtin type SwigPyPacked has no __module__ attribute
DeprecationWarning: builtin type SwigPyObject has no __module__ attribute
DeprecationWarning: builtin type swigvarlink has no __module__ attribute
```

These warnings are related to SWIG (Simplified Wrapper and Interface Generator) types used by FAISS. They don't affect functionality and are internal implementation details of the library.

## Troubleshooting

### Common Issues

1. **Test Case Loading Fails**
   - Check that JSON files are valid
   - Verify file paths in test configuration
   - Ensure test case structure matches expected format

2. **LLM Response Issues**
   - For local testing: Check mock responses in `utils.py`
   - For integration testing: Verify API credentials and connectivity

3. **Path-related Errors**
   - Ensure all paths in test configuration are relative to `tests/data/`
   - Check file permissions in the test directory

### Getting Help

If you encounter issues not covered here:

1. Check the test logs for detailed error messages
2. Review the test case configuration
3. Verify the orchestrator implementation
4. Contact the development team for assistance

## Best Practices

1. **Test Case Design**
   - Keep test cases focused and atomic
   - Test both happy paths and error cases
   - Include edge cases and boundary conditions
   - Document complex test scenarios

2. **Mock Responses**
   - Keep mock responses simple and predictable
   - Document any assumptions in the mock behavior
   - Update mocks when API behavior changes

3. **Maintenance**
   - Regularly update test cases as features evolve
   - Remove obsolete test cases
   - Keep documentation in sync with code changes

4. **Performance**
   - Use local testing for quick development cycles
   - Run integration tests before merging
   - Consider test execution time in CI/CD pipelines

5. **Documentation**
   - Keep test cases well-documented
   - Document any special test requirements
   - Update examples when adding new features
