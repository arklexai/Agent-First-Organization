from typing import List, Tuple, Optional
from arklex.utils.graph_state import ResourceRecord
from arklex.utils.model_provider_config import PROVIDER_EMBEDDINGS, PROVIDER_EMBEDDING_MODELS
from arklex.utils.model_config import MODEL
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import OpenAIEmbeddings
import numpy as np


class ShortTermMemory:
    def __init__(self, trajectory: List[List[ResourceRecord]]):
        """

        Args:
            trajectory (List[List[ResourceRecord]]): Memory structure for the conversation where 
                                                    each list of ResourceRecord objects encompasses 
                                                    the information of a single conversation turn
        """          
        self.trajectory = trajectory[-5:]  # Use the last 5 turns from the trajectory
        self.embedding_model_name = PROVIDER_EMBEDDING_MODELS[MODEL['llm_provider']]
        self.embedding_model = PROVIDER_EMBEDDINGS.get(MODEL['llm_provider'])(
            **{'model': self.embedding_model_name} if MODEL['llm_provider'] != 'anthropic' else {'model_name': self.embedding_model_name}
        )
    def retrieve_records(self, query: str, top_k: int = 3, threshold: float = 0.7) -> Tuple[bool, List[ResourceRecord]]:
        """

        Args:
            query (str):  The query string to retrieve relevant records for.
            top_k (int, optional): The number of top records to return. Defaults to 3.
            threshold (float, optional): The similarity score threshold for filtering relevant records. Defaults to 0.7.

        Returns:
            Tuple[bool, List[ResourceRecord]]: A tuple where the first element is a boolean indicating 
                                           whether relevant records were found, and the second element is a 
                                           list of the top-k relevant `ResourceRecord` objects based on the query.
        """
        if not self.trajectory:
            return False, []

        query_embedding = np.array(self.embedding_model.embed_query(query)).reshape(1, -1)
        scored_records = []
        total_turns = len(self.trajectory)

        # Loop through the trajectory and score the records
        for turn_idx, turn in enumerate(self.trajectory):
            recency_score = (turn_idx + 1) / total_turns

            for record in turn:
                score_components = {
                    "task": 0.0,
                    "intent": 0.0,
                    "context": 0.0,
                    "output": 0.0,
                    "recency": recency_score
                }

                task = record.info.get("attribute", {}).get("task")
                if task:
                    task_embedding = np.array(self.embedding_model.embed_query(task)).reshape(1, -1)
                    score_components["task"] = cosine_similarity(query_embedding, task_embedding)[0][0]

                if record.intent:
                    intent_embedding = np.array(self.embedding_model.embed_query(record.intent)).reshape(1, -1)
                    score_components["intent"] = cosine_similarity(query_embedding, intent_embedding)[0][0]

                for step in record.steps or []:
                    if isinstance(step, dict) and "context_generate" in step:
                        context_embedding = np.array(self.embedding_model.embed_query(step["context_generate"])).reshape(1, -1)
                        score_components["context"] = cosine_similarity(query_embedding, context_embedding)[0][0]
                        break

                if record.output:
                    output_embedding = np.array(self.embedding_model.embed_query(record.output)).reshape(1, -1)
                    score_components["output"] = cosine_similarity(query_embedding, output_embedding)[0][0]

                weighted_score = sum(score_components.values()) / len(score_components)
                scored_records.append({"record": record, "score": weighted_score})

        # Filter out the records that have a score below the threshold
        relevant_records = [r for r in scored_records if r["score"] >= threshold]
        if not relevant_records:
            return False, []

        # Sort the relevant records by score and return the top_k
        relevant_records.sort(key=lambda x: x["score"], reverse=True)
        return True, [r["record"] for r in relevant_records[:top_k]]

    def retrieve_intent(self, query: str, threshold: float = 0.7) -> Tuple[bool, Optional[str]]:
        """

        Args:
            query (str): The query string to retrieve the most relevant intent for.
            threshold (float, optional): The similarity score threshold for filtering relevant intents. Defaults to 0.7.

        Returns:
            Tuple[bool, Optional[str]]: A tuple where the first element is a boolean indicating 
                                     whether a relevant intent was found, and the second element is the 
                                     most relevant intent (if found), or None if no relevant intent meets 
                                     the threshold.
        """        
        if not self.trajectory:
            return False, None

        query_embedding = np.array(self.embedding_model.embed_query(query)).reshape(1, -1)
        best_intent = None
        best_score = -1.0

        # Loop through the recent trajectory to find the most relevant intent
        for turn in self.trajectory:
            for record in turn:
                if record.intent:
                    intent_embedding = np.array(self.embedding_model.embed_query(record.intent)).reshape(1, -1)
                    similarity = cosine_similarity(query_embedding, intent_embedding)[0][0]
                    if similarity > best_score:
                        best_score = similarity
                        best_intent = record.intent

        # If the best score is above the threshold, return the intent
        if best_score >= threshold:
            return True, best_intent
        else:
            return False, None
