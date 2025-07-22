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
> **Need to update**

The MAS is triggered via a Taskgraph configuration where a single node defines a `MultiAgent` and its behavior.

#### Example

```jsonc
{
  "id": "multi_agent",
  "name": "MultiAgent",
  "path": "multi_agent.py",
  "config": {
    "role": "agent_role",
    "pattern": "deterministic",  // orchestration logic
    "task": "Main task description",
    "is_async": false,           // async execution (optional)
    "sub_agents": [ ... ]        // list of participating agents
  }
}
```
> 💡 For async patterns like `parallel` and `llm_as_a_judge`, set `is_async`: true.
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


### Tool Configuration for Sub-Agents
Each tool used by a sub-agent should be configured like this:

```jsonc
"tools": [
  {
    "id": "get_user_details_admin",        // ID = name of the tool function
    "path": "shopify/get_user_details_admin.py", // Path to the module (relative to `arklex.env.tools`)
    "fixed_args": {                       // Optional: constant args passed at runtime
      "admin_token": "<shopify_admin_token>",
      "shop_url": "<your-shopify-shop-url>",
      "api_version": "2024-10"
    }
  }
]
```
- id: The name of the tool function (must match the Python function name).

- path: Path to the .py file that contains the tool (relative to your project structure). If null, the tool is considered built-in.

- fixed_args: Optional dictionary of arguments injected at runtime — useful for credentials or config that shouldn’t come from the user.

### Built-in OpenAI Tools
If the path is null, the tool is assumed to be built-in and will be looked up in this mapping:

```python
# Built-in tools mapping
BUILT_IN_TOOLS = {
    "web_search": WebSearchTool,
}
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
> **Need to update**

> Agents call domain tools like search or user info
    


```jsonc
    "agents": [
    {
        "id": "multi_agent",
        "name": "MultiAgent",
        "path": "multi_agent.py",
        "config": {
            "role": "shopify assistant",
            "pattern": "agents_as_tools",
            "task": "Help users find products and account info.",
            "sub_agents": [
                {
                    "name": "ProductSearchAgent",
                    "instructions": "Help the user search for products by keyword or category.",
                    "tools": [
                        {
                            "id": "search_products",
                            "path": "shopify/search_products.py",
                            "fixed_args": {
                                "admin_token": "<shopify_admin_token>",
                                    "shop_url": "<your-shopify-shop-url>",
                                "api_version": "2024-10",
                                "llm_provider":"openai",
                                "model_type_or_path":"gpt-4o-mini"
                            }
                        }
                    ]
                },
                {
                    "name": "UserInfoAgent",
                    "instructions": "Help the user retrieve detailed information about a customer by their ID. The user id, such as 'gid://shopify/Customer/13573257450893'",
                    "tools": [
                        {
                        "id": "get_user_details_admin",
                        "path": "shopify/get_user_details_admin.py",
                        "fixed_args": {
                            "admin_token": "<shopify_admin_token>",
                            "shop_url": "<your-shopify-shop-url>",
                            "api_version": "2024-10"
                        }
                        }
                    ]
                    }
            ]
            }
    }
    ]
```


## TODO: Taskgraph Generation
Add a ability to auto-generate `taskgraph.json` with Multi-Agent System support