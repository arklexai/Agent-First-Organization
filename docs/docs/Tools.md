# Tools

Tools are the building blocks of the Arklex framework, providing functionality for
various operations such as intent detection, slot filling, and task execution.

## Overview

Tools in the Arklex framework are designed to be modular and reusable components
that can be combined to create complex workflows. Each tool has a well-defined
interface for input and output. Powered through the slot filling mechanism, the
framework handles the translation between MessageState and the specified parameter
and return values, allowing smoother development of custom methods and integration
of API.

## Tool Structure

Each tool in the Arklex framework follows a standard structure:

```python
from arklex.env.tools.tools import BaseTool
from arklex.utils.graph_state import MessageState
from arklex.orchestrator.NLU.entities.slot_entities import Slot

class MyTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="my_tool",
            description="Description of my tool"
        )

    def _get_required_slots(self) -> List[Slot]:
        return [
            Slot(
                name="param1",
                description="Description of parameter 1",
                required=True,
                type="string"
            ),
            Slot(
                name="param2",
                description="Description of parameter 2",
                required=False,
                type="integer"
            )
        ]

    def _execute_impl(self, message_state: MessageState) -> Any:
        # Tool implementation
        return result
```

## Slot Definition

Slots are used to define the parameters that a tool requires. Each slot has the
following attributes:

- **`name`**: The name of the parameter
- **`type`**: The type of the parameter (string, integer, float, boolean)
- **`required`**: Whether the parameter is required
- **`enum`**: The candidate values for the slot. This is used to aid the slot
  filler to check if the extracted value is valid.
- **`description`**: A string describing the parameter. This will be used to aid
  the extraction component of slot filling. Adding examples often help the slot
  filling.

## Tool Registration

Tools can be registered using the `@tool` decorator:

```python
from arklex.env.tools.tools import tool

@tool(name="my_tool", description="Description of my tool")
def my_tool(param1: str, param2: int = 0) -> str:
    # Tool implementation
    return result
```

## Tool Execution

Tools are executed through the environment:

```python
from arklex.env.env import Environment

# Initialize environment
env = Environment(workers=[], tools=[my_tool], agents=[])

# Execute tool
result = env.execute_tool("my_tool", message_state)
```

## Best Practices

1. **Tool Design**
   - Keep tools focused and single-purpose
   - Use clear parameter names
   - Provide detailed descriptions
   - Handle errors gracefully

2. **Slot Definition**
   - Use clear slot names
   - Provide detailed descriptions
   - Include examples when helpful
   - Define appropriate types

3. **Error Handling**
   - Validate input parameters
   - Handle edge cases
   - Provide clear error messages
   - Log errors appropriately

4. **Testing**
   - Write unit tests
   - Test edge cases
   - Test error handling
   - Test integration

## Integration

Tools integrate with:

- Environment
- Message Processing
- Slot Filling
- State Management
- Logging System
