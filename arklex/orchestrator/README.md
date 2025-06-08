# Arklex Orchestrator

The Orchestrator is the main component of the AgentOrg framework, providing a task-oriented orchestration system that manages task completion through multiple turns of interaction. It is responsible for managing workers and executing tasks based on a graph-based workflow.

## Overview

The Orchestrator takes JSON/YAML configuration as input, reconstructs a task graph, and executes workers according to the graph structure. The system is designed to handle complex, multi-step tasks through a flexible and extensible architecture.

## Key Components

### Graph Structure

- **Nodes**: Represent workers that execute specific sub-tasks
- **Edges**: Represent user intents and define the flow between workers
- **Leaf Nodes**: Mark the completion of a task

### Worker Management

- Workers execute defined sub-tasks
- Each worker may require:
  - User input
  - Pre-defined values
  - Results from previous workers
- Workers return results that influence the next steps

### Task Handling

- Support for multiple concurrent tasks
- Task detection through NLU module
- Graph traversal based on user intents
- Result propagation between workers

## Workflow

1. **Task Detection**
   - NLU module parses user input
   - Detects intent and connected node
   - Initiates task execution

2. **Task Execution**
   - Starts from detected node
   - Traverses graph based on edges
   - Executes workers in sequence
   - Passes results to next worker

3. **Task Completion**
   - Reaches leaf node
   - Returns final results
   - Resets for next task

## Features

- Graph-based task management
- Multi-turn task execution
- Flexible worker configuration
- Intent-based navigation
- Result propagation
- Concurrent task handling

## Usage

```python
from arklex.orchestrator import Orchestrator

# Initialize orchestrator with configuration
orchestrator = Orchestrator(config_path="task_config.yaml")

# Process user input
response = orchestrator.process_input(user_input)

# Get task status
status = orchestrator.get_task_status()
```

## Configuration

The orchestrator accepts configuration in JSON or YAML format, defining:

- Task graph structure
- Worker definitions
- Edge conditions
- Input requirements
- Output specifications

## Dependencies

- NLU module for intent detection
- Worker implementations
- Graph management system
- Configuration parser
