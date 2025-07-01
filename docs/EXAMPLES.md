# 💡 Examples Guide

Complete examples and implementation patterns for Arklex AI agents across different use cases.

## 📋 Table of Contents

- [Customer Service Agent](#-customer-service-agent)
- [E-commerce Assistant](#-e-commerce-assistant)
- [Research Assistant](#-research-assistant)
- [Document Processor](#-document-processor)
- [Scheduling Assistant](#-scheduling-assistant)
- [Data Analyst](#-data-analyst)
- [Multi-Agent Workflow](#-multi-agent-workflow)
- [Custom Tools](#-custom-tools)

## 🎯 Customer Service Agent

A RAG-powered customer service agent with database memory and multi-step support workflows.

### Configuration

```json
{
  "name": "Customer Service Agent",
  "description": "RAG-powered support agent with database memory",
  "version": "1.0.0",
  "orchestrator": {
    "llm_provider": "openai",
    "model": "gpt-4o",
    "temperature": 0.7,
    "max_tokens": 1000
  },
  "workers": {
    "rag_worker": {
      "enabled": true,
      "vector_db": "milvus",
      "collection_name": "customer_support_docs"
    },
    "database_worker": {
      "enabled": true,
      "connection_string": "mysql://user:pass@localhost/arklex_db"
    }
  },
  "tools": ["calculator_tool", "email_tool"]
}
```

### Implementation

```python
from arklex import Orchestrator, TaskGraph
from arklex.workers import RAGWorker, DatabaseWorker
from arklex.tools import CalculatorTool, EmailTool

# Create orchestrator
orchestrator = Orchestrator(
    llm_provider="openai",
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Add workers
orchestrator.add_worker(RAGWorker(vector_db="milvus"))
orchestrator.add_worker(DatabaseWorker())

# Add tools
orchestrator.add_tool(CalculatorTool())
orchestrator.add_tool(EmailTool())

# Execute workflow
result = orchestrator.run(task_graph, query="How do I reset my password?")
print(result.response)
```

## 🛒 E-commerce Assistant

An intelligent e-commerce assistant that can handle orders, inventory, and customer inquiries.

### Configuration

```json
{
  "name": "E-commerce Assistant",
  "description": "Intelligent e-commerce order and inventory management",
  "version": "1.0.0",
  "orchestrator": {
    "llm_provider": "openai",
    "model": "gpt-4o",
    "temperature": 0.7
  },
  "workers": {
    "database_worker": {
      "enabled": true,
      "connection_string": "mysql://user:pass@localhost/ecommerce_db"
    }
  },
  "tools": ["shopify_tool", "calculator_tool", "email_tool"]
}
```

### Implementation

```python
from arklex.tools import ShopifyTool, CalculatorTool, EmailTool

# Create orchestrator
orchestrator = Orchestrator(
    llm_provider="openai",
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Add Shopify integration
orchestrator.add_tool(ShopifyTool(
    api_key=os.getenv("SHOPIFY_API_KEY"),
    store_url="your-store.myshopify.com"
))

orchestrator.add_tool(CalculatorTool())
orchestrator.add_tool(EmailTool())

# Handle order query
result = orchestrator.run(task_graph, query="I want to order 2 red t-shirts")
print(result.response)
```

## 🔬 Research Assistant

A research assistant that can search the web, analyze documents, and synthesize information.

### Configuration

```json
{
  "name": "Research Assistant",
  "description": "Intelligent research and document analysis",
  "version": "1.0.0",
  "orchestrator": {
    "llm_provider": "openai",
    "model": "gpt-4o",
    "temperature": 0.3,
    "max_tokens": 2000
  },
  "workers": {
    "rag_worker": {
      "enabled": true,
      "vector_db": "milvus",
      "collection_name": "research_documents"
    },
    "browser_worker": {
      "enabled": true,
      "headless": true,
      "timeout": 30
    }
  },
  "tools": ["web_search_tool", "calculator_tool", "document_processor_tool"]
}
```

### Implementation

```python
from arklex.tools import WebSearchTool, DocumentProcessorTool

# Create orchestrator
orchestrator = Orchestrator(
    llm_provider="openai",
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Add research tools
orchestrator.add_tool(WebSearchTool(api_key=os.getenv("TAVILY_API_KEY")))
orchestrator.add_tool(DocumentProcessorTool())

# Conduct research
result = orchestrator.run(task_graph, query="What are the latest developments in quantum computing?")
print(result.response)
```

## 📄 Document Processor

An intelligent document processor that can extract, analyze, and summarize information.

### Configuration

```json
{
  "name": "Document Processor",
  "description": "Intelligent document processing and analysis",
  "version": "1.0.0",
  "orchestrator": {
    "llm_provider": "openai",
    "model": "gpt-4o",
    "temperature": 0.3,
    "max_tokens": 1500
  },
  "workers": {
    "rag_worker": {
      "enabled": true,
      "vector_db": "milvus",
      "collection_name": "processed_documents"
    },
    "document_worker": {
      "enabled": true,
      "supported_formats": ["pdf", "docx", "txt", "csv"]
    }
  },
  "tools": ["document_processor_tool", "ocr_tool", "table_extractor_tool"]
}
```

### Implementation

```python
from arklex.tools import DocumentProcessorTool, OCRTool, TableExtractorTool

# Create orchestrator
orchestrator = Orchestrator(
    llm_provider="openai",
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Add document processing tools
orchestrator.add_tool(DocumentProcessorTool())
orchestrator.add_tool(OCRTool())
orchestrator.add_tool(TableExtractorTool())

# Process document
result = orchestrator.run(task_graph, query="Process this PDF document")
print(result.response)
```

## 📅 Scheduling Assistant

An intelligent scheduling assistant that can manage calendars and book appointments.

### Configuration

```json
{
  "name": "Scheduling Assistant",
  "description": "Intelligent calendar and appointment management",
  "version": "1.0.0",
  "orchestrator": {
    "llm_provider": "openai",
    "model": "gpt-4o",
    "temperature": 0.7
  },
  "workers": {
    "database_worker": {
      "enabled": true,
      "connection_string": "mysql://user:pass@localhost/scheduling_db"
    }
  },
  "tools": ["google_calendar_tool", "email_tool", "calculator_tool"]
}
```

### Implementation

```python
from arklex.tools import GoogleCalendarTool, EmailTool

# Create orchestrator
orchestrator = Orchestrator(
    llm_provider="openai",
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Add scheduling tools
orchestrator.add_tool(GoogleCalendarTool(
    credentials_file=os.getenv("GOOGLE_CALENDAR_CREDENTIALS")
))

orchestrator.add_tool(EmailTool())

# Handle scheduling request
result = orchestrator.run(task_graph, query="Schedule a meeting with John tomorrow at 2 PM")
print(result.response)
```

## 📊 Data Analyst

An intelligent data analyst that can process, analyze, and visualize data.

### Configuration

```json
{
  "name": "Data Analyst",
  "description": "Intelligent data analysis and visualization",
  "version": "1.0.0",
  "orchestrator": {
    "llm_provider": "openai",
    "model": "gpt-4o",
    "temperature": 0.3,
    "max_tokens": 2000
  },
  "workers": {
    "database_worker": {
      "enabled": true,
      "connection_string": "mysql://user:pass@localhost/analytics_db"
    },
    "data_worker": {
      "enabled": true,
      "supported_formats": ["csv", "json", "xlsx", "parquet"]
    }
  },
  "tools": ["calculator_tool", "chart_generator_tool", "statistical_analysis_tool"]
}
```

### Implementation

```python
from arklex.tools import ChartGeneratorTool, StatisticalAnalysisTool

# Create orchestrator
orchestrator = Orchestrator(
    llm_provider="openai",
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Add analysis tools
orchestrator.add_tool(ChartGeneratorTool())
orchestrator.add_tool(StatisticalAnalysisTool())

# Analyze data
result = orchestrator.run(task_graph, query="Analyze sales data for Q4 2023")
print(result.response)
```

## 🤖 Multi-Agent Workflow

A complex workflow involving multiple specialized agents working together.

### Configuration

```json
{
  "name": "Multi-Agent Workflow",
  "description": "Complex workflow with multiple specialized agents",
  "version": "1.0.0",
  "agents": {
    "research_agent": {
      "orchestrator": {
        "llm_provider": "openai",
        "model": "gpt-4o",
        "temperature": 0.3
      },
      "tools": ["web_search_tool", "document_processor_tool"]
    },
    "analysis_agent": {
      "orchestrator": {
        "llm_provider": "openai",
        "model": "gpt-4o",
        "temperature": 0.3
      },
      "tools": ["calculator_tool", "statistical_analysis_tool"]
    },
    "writing_agent": {
      "orchestrator": {
        "llm_provider": "openai",
        "model": "gpt-4o",
        "temperature": 0.7
      },
      "tools": ["document_processor_tool"]
    }
  }
}
```

### Implementation

```python
class MultiAgentWorkflow:
    def __init__(self, config):
        self.agents = {}
        
        # Initialize agents
        for agent_name, agent_config in config['agents'].items():
            self.agents[agent_name] = Orchestrator(**agent_config['orchestrator'])
            
            # Add tools to agent
            for tool_name in agent_config['tools']:
                tool = self.create_tool(tool_name)
                self.agents[agent_name].add_tool(tool)
    
    def execute_workflow(self, task):
        """Execute multi-agent workflow"""
        # Step 1: Research phase
        research_results = self.agents['research_agent'].run(
            self.create_research_task_graph(),
            query=task['research_query']
        )
        
        # Step 2: Analysis phase
        analysis_results = self.agents['analysis_agent'].run(
            self.create_analysis_task_graph(),
            query=task['analysis_query'],
            context={'research_data': research_results}
        )
        
        # Step 3: Writing phase
        final_report = self.agents['writing_agent'].run(
            self.create_writing_task_graph(),
            query=task['writing_query'],
            context={
                'research_data': research_results,
                'analysis_data': analysis_results
            }
        )
        
        return final_report

# Usage
workflow = MultiAgentWorkflow(config)
result = workflow.execute_workflow({
    'research_query': 'Research market trends in AI',
    'analysis_query': 'Analyze the research data',
    'writing_query': 'Write a comprehensive report'
})
```

## 🛠️ Custom Tools

Examples of custom tools that extend Arklex AI's capabilities.

### Weather Tool

```python
from arklex.tools import BaseTool
import requests

class WeatherTool(BaseTool):
    def __init__(self, api_key):
        super().__init__()
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5"
    
    def execute(self, input_data):
        """Get weather information for a location"""
        location = input_data.get('location')
        
        if not location:
            return {"error": "Location is required"}
        
        try:
            response = requests.get(
                f"{self.base_url}/weather",
                params={
                    'q': location,
                    'appid': self.api_key,
                    'units': 'metric'
                }
            )
            
            if response.status_code == 200:
                weather_data = response.json()
                return {
                    "success": True,
                    "data": {
                        "location": location,
                        "temperature": weather_data['main']['temp'],
                        "description": weather_data['weather'][0]['description']
                    }
                }
            else:
                return {"error": f"Weather API error: {response.status_code}"}
                
        except Exception as e:
            return {"error": f"Weather tool error: {str(e)}"}
```

### Translation Tool

```python
from arklex.tools import BaseTool
from googletrans import Translator

class TranslationTool(BaseTool):
    def __init__(self):
        super().__init__()
        self.translator = Translator()
    
    def execute(self, input_data):
        """Translate text between languages"""
        text = input_data.get('text')
        target_language = input_data.get('target_language', 'en')
        
        if not text:
            return {"error": "Text is required"}
        
        try:
            translation = self.translator.translate(text, dest=target_language)
            
            return {
                "success": True,
                "data": {
                    "original_text": text,
                    "translated_text": translation.text,
                    "target_language": translation.dest
                }
            }
            
        except Exception as e:
            return {"error": f"Translation error: {str(e)}"}
```

---

For more examples and implementation details, see the [API Reference](API.md) and [Quick Start Guide](QUICKSTART.md).
