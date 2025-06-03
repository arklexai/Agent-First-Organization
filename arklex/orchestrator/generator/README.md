# Arklex Task Graph Generator

The Task Graph Generator is a component of the Arklex framework that automatically generates orchestrator configuration files (TaskGraph) based on natural language descriptions and domain knowledge. It enables users to define complex tasks and their required resources through natural language, which are then converted into structured configuration files for the orchestrator.

## Overview

The generator takes two main inputs:

1. Task Description: Natural language description of the task to be performed
2. Domain Knowledge: Information about available resources and capabilities

These inputs are processed to generate a JSON configuration file that defines the task graph structure, including nodes, edges, and resource requirements.

## Key Features

- Natural Language Task Definition
- Domain Knowledge Integration
- Resource Mapping
- Configuration Generation
- Customization Support

## Input Components

### Task Description

- High-level task goals
- Required steps and actions
- Expected outcomes
- User interaction points

### Domain Knowledge

- Available APIs and endpoints
- Database schemas and access
- Website content and structure
- External service integrations
- Custom tools and utilities

## Output

The generator produces a JSON configuration file that includes:

- Task graph structure
- Node definitions
- Edge conditions
- Resource mappings
- Input/output specifications
- Error handling rules

## Usage

```python
from arklex.orchestrator.generator import TaskGraphGenerator

# Initialize generator
generator = TaskGraphGenerator()

# Generate configuration
config = generator.generate(
    task_description="Process customer orders and update inventory",
    domain_knowledge={
        "apis": ["order_api", "inventory_api"],
        "databases": ["customer_db", "product_db"],
        "tools": ["email_sender", "notification_service"]
    }
)

# Save configuration
generator.save_config(config, "task_graph.json")
```

## Customization

The generated configuration can be customized to:

- Modify task flow
- Add custom nodes
- Adjust resource mappings
- Define specific conditions
- Add error handling
- Configure timeouts

## Integration

The generated configuration is used by the orchestrator to:

1. Reconstruct the task graph
2. Initialize required resources
3. Set up worker nodes
4. Configure edge conditions
5. Establish monitoring points

## Best Practices

1. Provide clear and detailed task descriptions
2. Include comprehensive domain knowledge
3. Review generated configuration
4. Test with sample inputs
5. Iterate and refine as needed
