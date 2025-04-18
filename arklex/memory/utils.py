from typing import List, Union
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import OpenAIEmbeddings
from arklex.utils.graph_state import ResourceRecord, MessageState
import numpy as np

def get_relevant_records(state: MessageState, query: str, top_k: int = 3, threshold: float = 0.7) -> Union[List[ResourceRecord], str]:
    if not state.trajectory:
        return "not in context"

    embeddings = OpenAIEmbeddings()
    query_embedding = np.array(embeddings.embed_query(query)).reshape(1, -1)
    
    scored_records = []
    recent_turns = state.trajectory[-5:]
    total_turns = len(recent_turns)

    for turn_idx, turn in enumerate(recent_turns):
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
                try:
                    task_embedding = np.array(embeddings.embed_query(task)).reshape(1, -1)
                    score_components["task"] = cosine_similarity(query_embedding, task_embedding)[0][0]
                except:
                    pass

            if record.intent:
                try:
                    intent_embedding = np.array(embeddings.embed_query(record.intent)).reshape(1, -1)
                    score_components["intent"] = cosine_similarity(query_embedding, intent_embedding)[0][0]
                except:
                    pass

            for step in record.steps or []:
                if isinstance(step, dict) and "context_generate" in step:
                    try:
                        context_embedding = np.array(embeddings.embed_query(step["context_generate"])).reshape(1, -1)
                        score_components["context"] = cosine_similarity(query_embedding, context_embedding)[0][0]
                    except:
                        pass
                    break

            if record.output:
                try:
                    output_embedding = np.array(embeddings.embed_query(record.output)).reshape(1, -1)
                    score_components["output"] = cosine_similarity(query_embedding, output_embedding)[0][0]
                except:
                    pass

            weighted_score = sum(score_components.values()) / len(score_components)
            scored_records.append({"record": record, "score": weighted_score})

    relevant_records = [r for r in scored_records if r["score"] >= threshold]
    if not relevant_records:
        return "not in context"

    relevant_records.sort(key=lambda x: x["score"], reverse=True)
    return [r["record"] for r in relevant_records[:top_k]]

def get_most_relevant_intent(state: MessageState, query: str, threshold: float = 0.7) -> Union[str, None]:
    if not state.trajectory:
        return "not in context"

    embeddings = OpenAIEmbeddings()
    query_embedding = np.array(embeddings.embed_query(query)).reshape(1, -1)

    best_intent = None
    best_score = -1.0

    recent_turns = state.trajectory[-5:]

    for turn in recent_turns:
        for record in turn:
            if record.intent:
                try:
                    intent_embedding = np.array(embeddings.embed_query(record.intent)).reshape(1, -1)
                    similarity = cosine_similarity(query_embedding, intent_embedding)[0][0]
                    if similarity > best_score:
                        best_score = similarity
                        best_intent = record.intent
                except:
                    continue

    return best_intent if best_score >= threshold else "not in context"
