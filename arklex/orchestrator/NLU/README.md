# Natural Language Understanding (NLU)

The NLU module provides functionality for natural language understanding, including
intent detection, slot filling, and entity recognition.

## Overview

The NLU module is designed to be modular and extensible, with a clear separation of
concerns between different components. It provides a unified interface for natural
language understanding tasks through the `SlotFiller` class.

## Components

### Core

The core components provide the fundamental functionality for natural language
understanding:

- `SlotFiller`: Main class for slot filling operations
- `IntentDetector`: Class for intent detection
- `EntityRecognizer`: Class for entity recognition

### API

The API components provide HTTP endpoints for the NLU functionality:

```python
from arklex.orchestrator.NLU.core.routes import create_app

app = create_app()
```

## Usage

### Slot Filling

```python
from arklex.orchestrator.NLU.core.slot import SlotFiller
from arklex.utils.slot import Slot

# Initialize slot filler
slot_filler = SlotFiller(url="http://localhost:8000")

# Define slots
slots = [
    Slot(
        name="name",
        description="User's name",
        required=True,
        type="string"
    ),
    Slot(
        name="age",
        description="User's age",
        required=False,
        type="integer"
    )
]

# Fill slots
filled_slots = slot_filler.execute(slots, "My name is John and I am 30 years old")
```

### Intent Detection

```python
from arklex.orchestrator.NLU.core.intent import IntentDetector

# Initialize intent detector
intent_detector = IntentDetector(url="http://localhost:8000")

# Detect intent
intent = intent_detector.execute("I want to book a flight to New York")
```

### Entity Recognition

```python
from arklex.orchestrator.NLU.core.entity import EntityRecognizer

# Initialize entity recognizer
entity_recognizer = EntityRecognizer(url="http://localhost:8000")

# Recognize entities
entities = entity_recognizer.execute("I want to book a flight to New York")
```

## Configuration

The NLU module can be configured through environment variables or a configuration file:

```python
from arklex.orchestrator.NLU.core.config import Config

config = Config(
    api_url="http://localhost:8000",
    model_path="models/nlu",
    batch_size=32
)
```

## Best Practices

1. **Error Handling**
   - Handle API errors gracefully
   - Provide meaningful error messages
   - Log errors appropriately

2. **Performance**
   - Use batch processing when possible
   - Cache results when appropriate
   - Monitor resource usage

3. **Testing**
   - Write unit tests for each component
   - Test edge cases
   - Test error handling
   - Test integration

4. **Documentation**
   - Document API endpoints
   - Document configuration options
   - Document best practices
   - Document examples

## Integration

The NLU module integrates with:

- Environment
- Message Processing
- State Management
- Logging System
