import logging
import requests
from typing import Dict, Any, Optional

from langgraph.graph import StateGraph, START
from langchain_openai import ChatOpenAI

from arklex.env.workers.worker import BaseWorker, register_worker
from arklex.utils.graph_state import MessageState
from arklex.utils.model_config import MODEL

logger = logging.getLogger(__name__)

@register_worker
class HTTPWorker(BaseWorker):
    description = "Make HTTP requests to external APIs and handle responses"

    def __init__(self, **kwargs):
        super().__init__()
        self.endpoint = kwargs.get('endpoint')
        self.method = kwargs.get('method', 'GET').upper()
        self.headers = kwargs.get('headers', {'Content-Type': 'application/json'})
        self.body = kwargs.get('body', None)
        self.query_params = kwargs.get('query_params', {})
        logger.info(f"HTTPWorker initialized with endpoint: {self.endpoint}, method: {self.method}, headers: {self.headers}")
        self.action_graph = self._create_action_graph()

    def make_request(self, state: MessageState) -> MessageState:
        """Make the HTTP request with the configured parameters"""
        try:
            endpoint = self.endpoint
            method = self.method
            headers = self.headers
            body = self.body
            query_params = self.query_params
                
            logger.info(f"Making {method} request to {endpoint}")
            logger.info(f"Headers: {headers}")
            logger.info(f"Query params: {query_params}")
            logger.info(f"Body: {body}")
            
            response = requests.request(
                method=method,
                url=endpoint,
                headers=headers,
                json=body,
                params=query_params
            )
            response.raise_for_status()
            
            response_data = response.json()
            state.response = str(response_data)  
            state.message_flow = f"Successfully made {method} request to {endpoint}"
            logger.info(f"Request successful. Response: {state.response}")
            return state
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error making HTTP request: {str(e)}")
            state.message_flow = f"Error making HTTP request: {str(e)}"
            state.response = None
            return state
        except Exception as e:
            logger.error(f"Unexpected error in HTTPWorker: {str(e)}")
            state.message_flow = f"Unexpected error: {str(e)}"
            state.response = None
            return state

    def _create_action_graph(self):
        workflow = StateGraph(MessageState)
        workflow.add_node("make_request", self.make_request)
        workflow.add_edge(START, "make_request")
        return workflow

    def _execute(self, msg_state: MessageState, **kwargs):
        graph = self.action_graph.compile()
        result = graph.invoke(msg_state)
        return result 