import logging
import requests
import inspect

from arklex.env.tools.tools import Tool, register_tool
from arklex.utils.graph_state import HTTPParams
from arklex.exceptions import ToolExecutionError

logger = logging.getLogger(__name__)

@register_tool(
    desc="Make HTTP requests to external APIs and handle responses",
    slots=[],
    outputs=["response"],
    isResponse=False
)
def http_tool(**kwargs) -> str:
    """Make an HTTP request and return the response"""
    func_name = inspect.currentframe().f_code.co_name
    try:
        params = HTTPParams(**kwargs)
        logger.info(f"Making a {params.method} request to {params.endpoint}, with body: {params.body} and params: {params.params}")
        response = requests.request(
            method=params.method,
            url=params.endpoint,
            headers=params.headers,
            json=params.body,
            params=params.params
        )
        response.raise_for_status()
        response_data = response.json()
        logger.info(f"Response from the {params.endpoint} for body: {params.body} and params: {params.params} is: {response_data}")
        return str(response_data)
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error making HTTP request: {str(e)}")
        raise ToolExecutionError(func_name, f"Error making HTTP request: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in HTTPTool: {str(e)}")
        raise ToolExecutionError(func_name, f"Unexpected error: {str(e)}")

http_tool.__name__ = "http_tool" 