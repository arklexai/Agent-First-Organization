# Arklex Orchestrator Examples

This directory contains pre-defined orchestrator configurations for various use cases. Each example demonstrates how to structure and configure the Arklex orchestrator for specific domains and purposes.

## Available Examples

### E-commerce Assistant

- File: `ecommerce_assistant.json`
- Purpose: Customer service and product support
- Features:
  - Product search and discovery
  - Order management
  - Returns and exchanges
  - Customer support

### Healthcare Screener

- File: `healthcare_screener.json`
- Purpose: Medical screening and assessment
- Features:
  - Patient intake
  - Symptom assessment
  - Risk evaluation
  - Medical history collection

### Unemployment Government Assistant

- File: `unemployment_gov_assistant.json`
- Purpose: Government benefits support
- Features:
  - Eligibility checking
  - Application assistance
  - Status tracking
  - Document submission

### Advanced Roleplay

- File: `adv_roleplay.json`
- Purpose: Interactive roleplay scenarios
- Features:
  - Character interaction
  - Scenario management
  - Response generation
  - Context maintenance

## Usage

Each example configuration can be used as a template for creating new orchestrators. The JSON files contain:

- Task graph structure
- Node definitions
- Edge conditions
- Resource mappings
- Input/output specifications

To use an example:

1. Copy the desired JSON file
2. Modify the configuration to match your requirements
3. Load the configuration in your orchestrator

Example:

```python
from arklex.orchestrator import Orchestrator

# Load example configuration
with open("ecommerce_assistant.json", "r") as f:
    config = json.load(f)

# Initialize orchestrator
orchestrator = Orchestrator(config)

# Start processing
orchestrator.process()
```

## Customization

The example configurations can be customized by:

- Modifying task definitions
- Adding new nodes
- Adjusting edge conditions
- Updating resource mappings
- Changing input/output formats

## Best Practices

1. Start with a similar example to your use case
2. Review the task graph structure
3. Understand the resource mappings
4. Test with sample inputs
5. Iterate and refine as needed
