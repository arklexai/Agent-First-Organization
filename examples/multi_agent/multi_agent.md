# 🧠 Multi-Agent System (MAS)

## Overview

The **Multi-Agent System (MAS)** enables orchestration of multiple specialized agents to collaborate on complex tasks through configurable execution patterns. It allows defining, composing, and running agent pipelines dynamically using a flexible config-driven architecture.

### 🔧 Features
- 🧩 Modular agent definitions 
- 🧠 Pattern-based orchestration (`deterministic`, `agents_as_tools`)
- 📦 Configurable via JSON 

---

## Taskgraph

The MAS can be triggered through a **Taskgraph configuration**, where the multi-agent pipeline is defined as a single `MultiAgent` node.

The agent is initialized with a `taskgraph` containing `agents` field with:
- `pattern`: The orchestration logic (e.g., `deterministic`, `agents_as_tools`)
- `sub_agents`: The sub-agents that will be used in the MAS

---

## 🛠 How Tools Work
Each sub-agent in a multi-agent configuration can optionally use one or more tools to perform specialized tasks (e.g., search Shopify, retrieve user info, search the web). 
### 🔗 Tool Types
A tool can be defined in one of the following ways:

- **Regular Python function**

    Automatically wrapped as a `FunctionTool` when used in a sub-agent.

- **`FunctionTool` instance**

    Explicitly wrapped function using the `FunctionTool` class.

- **Built-in OpenAI Agent SDK tool**

    Tools like `web_search` that require no path; just reference them by id.

- **Arklex-defined tool (e.g., Shopify)**

    Domain-specific tools like search_products or get_user_details_admin.
    These are automatically converted to `FunctionTool` and can accept fixed_args (e.g., API credentials).

>💡 For Arklex tools, pass required credentials or config as `fixed_args` in your sub-agent definition. These will be injected at runtime, so the function does not need to read them from environment variables.


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
## Agent Portion of Taskgraph Examples
### 🔍 Research Assistant

Goal: Help a user investigate a topic using real-time search and generate a detailed summary.

- **Pattern: `deterministic`**

    Sub-Agents: Planner → Search → Writer
    ```jsonc
    "agents": [
    {
        "id": "multi_agent",
        "name": "MultiAgent",
        "path": "multi_agent.py",
        "config": {
            "role": "research_bot",
            "pattern": "deterministic",
            "task": "Help the user research a topic using a multi-step agent pipeline.",
            "sub_agents": [
                {
                    "name": "PlannerAgent",
                    "instructions": "Break down the research topic into a list of relevant web search queries. BE CONCISE. Limit to 5 steps."
                },
                {
                    "name": "SearchAgent",
                    "instructions": "Search the web and summarize findings.",
                     "tools": [{"id": "web_search", "path": null}]
                },
                {
                    "name": "WriterAgent",
                    "instructions": "Combine the search results into a coherent and informative report on the original research topic. Include links from previous step if appropiate"
                }
    ]
        }
    }
    ]
    ```
    - **Pattern: `agents_as_tools`**

        Sub-Agents: Orchestrator → Search or/and CitationFinder
        ```jsonc
        "agents": [
        {
            "id": "multi_agent",
            "name": "MultiAgent",
            "path": "multi_agent.py",
            "config": {
                "role": "research_bot",
                "pattern": "agents_as_tools",
                "task": "Research a topic by combining real-time web summaries with academic citations for credibility and depth.",
                "sub_agents": [
                        {
                            "name": "SearchAgent",
                            "instructions": "Search the web and summarize findings. Make sure to search for the most up to date information",
                            "tools": [{"id": "web_search", "path": null}]
                        },
                        {
                            "name": "CitationFinderAgent",
                            "instructions": "You receive a paragraph or claim and return one or more credible sources (academic or journalistic) that support the content.",
                            "tools": [{"id": "citation_finder", "path": "multi_agent/citation_finder.py"}]
                        }
                    ]
                }
        }
        ]
    ```
### 🛒 Shopify Assistant

Goal: Help users find products and account info.
    
- **Pattern: `agents_as_tools`**

    Sub-Agents: Orchestrator → ProductSearch or/and UserInfo
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