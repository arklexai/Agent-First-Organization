import logging
import requests
from typing import Dict, Any, Optional

from arklex.env.tools.tools import Tool, register_tool
from arklex.utils.graph_state import MessageState, HTTPParams

logger = logging.getLogger(__name__)

@register_tool(
    desc="Make HTTP requests to external APIs and handle responses",
    slots=[],
    outputs=["response"],
    isResponse=True
)
def http_tool(**kwargs) -> str:
    """Make an HTTP request and return the response"""
    try:
        params = HTTPParams(**kwargs)
        
        logger.info(f"Making a {params.method} request to {params.endpoint}")
        response = requests.request(
            method=params.method,
            url=params.endpoint,
            headers=params.headers,
            json=params.body
        )
        response.raise_for_status()
        
        response_data = response.json()
        return str(response_data)
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error making HTTP request: {str(e)}")
        return f"Error making HTTP request: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error in HTTPTool: {str(e)}")
        return f"Unexpected error: {str(e)}"

# Register the tool with a specific name
http_tool.__name__ = "http_tool" 