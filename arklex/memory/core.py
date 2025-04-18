from typing import List, Union
from arklex.utils.graph_state import ResourceRecord, MessageState
from .utils import get_relevant_records, get_most_relevant_intent

class ShortTermMemory:
    def __init__(self, state: MessageState):
        self.state = state

    def retrieve_records(self, query: str, top_k: int = 3, threshold: float = 0.7) -> Union[List[ResourceRecord], str]:
        return get_relevant_records(self.state, query, top_k, threshold)

    def retrieve_intent(self, query: str, threshold: float = 0.7) -> Union[str, None]:
        return get_most_relevant_intent(self.state, query, threshold)