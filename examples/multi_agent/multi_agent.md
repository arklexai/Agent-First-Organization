# 🧠 Multi-Agent System (MAS)

## Overview

The **Multi-Agent System (MAS)** enables orchestration of multiple specialized agents to collaborate on complex tasks through configurable execution patterns. It allows defining, composing, and running agent pipelines dynamically using a flexible config-driven architecture.

---
### 🔧 Key Features
- 🧩 Modular agent definitions  
- 🧠 Pattern-based orchestration:
    - `agents_as_tools`
  - `deterministic` 
  - `parallel`
  - `llm_as_a_judge`
- 📦 JSON-configurable via Taskgraph
- 🔁 Async/sync support via `is_async` flag
> 💡 Note: `deterministic`,  `parallel`, and `llm_as_a_judge` patterns are still in development
---

## Configuration Format: Taskgraph
The MAS is triggered via a Taskgraph configuration where a single node defines a `MultiAgent` and its behavior.

#### Example `agents` field

```jsonc
 "agents": [
        {
            "id": "multi_agent",
            "name": "MultiAgent",
            "path": "multi_agent.py",
            "tools":[]
        },
        {
            "id": "openai_sdk_agent",
            "name": "OpenAISDKAgent",
            "path": null
        }
       
    ]
```
#### Example MultiAgent node
```jsonc
"nodes": [
        [...],           
        [
            "1",
            {
                "resource": {
                    "id": "multi_agent",
                    "name": "MultiAgent"
                },
                "attribute": {
                    "value": "",
                    "type": "agent", 
                    "task": "<Task instructions for multi-agent system>",
                    "direct": false,
                    "node_specific_data": {
                        "type": "agents_as_tools", // orchestration pattern
                        "is_async": true  // async execution (optional)
                    }
                }
            }
        ],

```
> 💡 For async patterns like `agents_as_tools`, `parallel` and `llm_as_a_judge`, set `is_async` to `true`.
---

## 🛠 Tooling Support
Each agent can use tools to perform specialized functions like web search, product lookup, etc.
### 🔗 Tool Types
A tool can be defined in one of the following ways:

- **Python function** — auto-wrapped as `FunctionTool`

- **Explicit `FunctionTool` instance**

- **Built-in OpenAI Agent SDK tool**

    Tools like `web_search` that require no path; just reference them by id.

- **Arklex-defined tool (e.g., Shopify)**

    Domain-specific tools like search_products or get_user_details_admin.
    These are automatically converted to `FunctionTool` and can accept fixed_args (e.g., API credentials).

>💡 Pass fixed_args for secrets/config (e.g., API tokens) — no need for agents to read env vars directly.


### Tool Configuration (Multi-Agent Compatible)

Each tool used by a sub-agent should follow this format:

```jsonc
"tools": [
  {
    "id": "shopify/get_user_details_admin",         // Must match the tool's resource_id from the chatbots repo
    "name": "get_user_details_admin",               // Exact name of the function to load
    "path": "shopify/get_user_details_admin.py",    // Relative to arklex.env.tools (null if built-in)
    "fixed_args": {                                 // Optional runtime constants (e.g. API keys)
      "admin_token": "<shopify_admin_token>",
      "shop_url": "<your-shopify-url>",
      "api_version": "2024-10"
    }
  }
]
```
**Field Breakdown:**
- id – Unique resource_id of the tool (must align with resources defined in `chatbots` repo).

- name – Function name defined inside the tool module.

- path – Relative path to the .py file; set to null for built-in tools.

- fixed_args – Optional constants injected at runtime (e.g., credentials or config).


### Built-in OpenAI Tools
If the path is null, the tool is assumed to be built-in and will be looked up in this mapping:

```python
# Built-in tools mapping
BUILT_IN_TOOLS = {
    "web_search": WebSearchTool,
}
```
Note: you should add built-in-tool as follows to   `tools` field:
```jsonc
"tools": [
  {
    "id": "web_search",
    "name": "web_search"
    "path": null   
  }
]
```
---
## 🧱 Adding a New Pattern
### Step 1: Create a Pattern Class
```python
# arklex/env/agents/patterns/my_pattern.py
from arklex.env.agents.patterns.base_pattern import BasePattern
from langgraph.graph import StateGraph
from arklex.orchestrator.entities.msg_state_entities import MessageState

class MyNewPattern(BasePattern):
    async def step_fn(self, state: MessageState) -> MessageState:
        # your pattern logic
        return state
```
### Step 2: Register it
```python
# arklex/env/agents/patterns/registry.py

from arklex.env.agents.patterns.my_pattern import MyNewPattern

PATTERN_DISPATCHER = {
    "my_pattern": MyNewPattern,
    ...
}
```
---
##  Pattern Examples
> **Note: Only `agents_as_tools` pattern is supported as of (2025-07-22)**

### 🛒 Shopify Assistant — `agents_as_tools`

> The orchestrator agent (created behind the scenes) delegates tasks to tool-wrapped sub-agents (`sub_agents`), each with their own tools.

#### 📌 Important Notes
✅ To ensure the multi-agent workflow functions correctly:

1. Add edges from the MultiAgent node to each sub_agent (e.g. OpenAISDKAgent).

2. Add edges from the tool(s) (nodes of type tool) to the sub_agent. Ensure that the tools are predecessor nodes of the sub_agent that will use them.

3. List the tools explicitly in the `tools` field

Without these connections, the orchestrator won’t be able to discover or use sub-agents and their tools.

#### 🧠 agents field (Example)
```jsonc
 "agents": [
        {
            "id": "multi_agent",
            "name": "MultiAgent",
            "path": "multi_agent.py",
            "tools":[], // Orchestrator does not have tools directly
            "sub_agents": ["ProductSearchAgent", "UserInfoAgent"]

        }
       
]
```
#### 🗺 nodes (Partial Example)
```jsonc
[
  // MultiAgent node
  [
    "1",
    {
      "resource": {
        "id": "multi_agent",
        "name": "MultiAgent"
      },
      "attribute": {
        "value": "",
        "type": "agent",
        "task": "Help users find products and account info.",
        "direct": false,
        "node_specific_data": {
          "type": "agents_as_tools",
          "is_async": true
        }
      }
    }
  ],

  // Sub-agent 1: ProductSearchAgent
  [
    "2",
    {
      "resource": {
        "id": "openai_sdk_agent",
        "name": "OpenAISDKAgent"
      },
      "attribute": {
        "value": "",
        "type": "agent",
        "task": "Help the user search for products by keyword or category.",
        "direct": false,
        "node_specific_data": {
          "name": "ProductSearchAgent"
        }
      }
    }
  ],

  // Sub-agent 2: UserInfoAgent
  [
    "4",
    {
      "resource": {
        "id": "openai_sdk_agent",
        "name": "OpenAISDKAgent"
      },
      "attribute": {
        "value": "",
        "type": "agent",
        "task": "Help the user retrieve detailed information about a customer by their ID.",
        "direct": false,
        "node_specific_data": {
          "name": "UserInfoAgent"
        }
      }
    }
  ]

  // Add tool nodes here as well, and connect them via edges to the relevant sub-agent
]

```


## TODO: Taskgraph Generation
Add a ability to auto-generate `taskgraph.json` with Multi-Agent System support