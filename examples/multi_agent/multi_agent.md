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



### Examples
🔍 Research Assistant
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
                    "tools": ["web_search"]
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
                        "instructions": "Search the web and summarize findings. Make sure to search for the most up to date information"
                    },
                    {
                        "name": "CitationFinderAgent",
                        "instructions": "You receive a paragraph or claim and return one or more credible sources (academic or journalistic) that support the content.",
                        "tools": ["citation_finder"]
                    }
                ]
            }
        }
        ]

## TODO: Taskgraph Generation
Add a ability to auto-generate `taskgraph.json` with Multi-Agent System support