# Arklex Natural Language Understanding (NLU)

This directory contains the Natural Language Understanding (NLU) components of the Arklex framework. The NLU system is responsible for intent detection, slot filling, and understanding user input in conversations.

## Components

### Core NLU (`nlu.py`)

- Intent detection and classification
- Slot filling and value extraction
- Input processing and validation
- Response formatting
- Error handling

### API Layer (`api.py`)

- FastAPI endpoints for NLU services
- Model interaction and response generation
- Input/output formatting
- Error handling and validation
- Configuration management

## Features

### Intent Detection

- Multi-intent classification
- Confidence scoring
- Context-aware processing
- Fallback handling
- Custom intent support

### Slot Filling

- Entity extraction
- Value validation
- Type conversion
- Required slot handling
- Slot verification

### API Integration

- RESTful endpoints
- Streaming support
- Authentication
- Rate limiting
- Error responses

## Usage

### Core NLU

```python
from arklex.orchestrator.NLU.nlu import NLU, SlotFilling

# Initialize NLU
nlu = NLU(config={
    "model": "gpt-4",
    "api_key": "your-api-key"
})

# Detect intent
intent = nlu.execute("I want to book a flight")

# Initialize slot filling
slot_filler = SlotFilling(config={
    "model": "gpt-4",
    "api_key": "your-api-key"
})

# Fill slots
slots = slot_filler.execute(
    "I want to fly to New York on Monday",
    required_slots=["destination", "date"]
)
```

### API Endpoints

```python
from fastapi import FastAPI
from arklex.orchestrator.NLU.api import NLUModelAPI, SlotFillModelAPI

app = FastAPI()

# Initialize APIs
nlu_api = NLUModelAPI()
slot_api = SlotFillModelAPI()

# Add endpoints
app.post("/nlu/predict")(nlu_api.predict)
app.post("/slotfill/predict")(slot_api.predict)
app.post("/slotfill/verify")(slot_api.verify)
```

## Configuration

### NLU Configuration

```json
{
    "model": "gpt-4",
    "api_key": "your-api-key",
    "temperature": 0.7,
    "max_tokens": 100,
    "timeout": 30
}
```

### Slot Filling Configuration

```json
{
    "model": "gpt-4",
    "api_key": "your-api-key",
    "temperature": 0.7,
    "max_tokens": 100,
    "timeout": 30,
    "required_slots": ["destination", "date"]
}
```

## Best Practices

1. **Intent Detection**
   - Use clear and specific intent names
   - Include fallback intents
   - Consider context in classification
   - Validate intent confidence

2. **Slot Filling**
   - Define clear slot types
   - Validate slot values
   - Handle missing slots
   - Provide clear prompts

3. **API Usage**
   - Implement proper error handling
   - Use appropriate timeouts
   - Validate input/output
   - Monitor API usage

4. **Performance**
   - Cache common intents
   - Batch process when possible
   - Optimize model parameters
   - Monitor response times

## Error Handling

The NLU system handles various error cases:

- Invalid input format
- Missing required slots
- Invalid slot values
- API timeouts
- Model errors
- Configuration issues

## Integration

The NLU system integrates with:

- Task Graph
- Orchestrator
- Message Processing
- State Management
- Resource Management
