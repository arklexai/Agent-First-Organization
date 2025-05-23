# TODO(christian): fix annotations in this file.

import asyncio
import re
from typing import List, Tuple, Optional

import numpy as np
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

from arklex.memory.prompts import intro, examples, output_instructions
from arklex.utils.graph_state import ResourceRecord, LLMConfig
from arklex.utils.model_provider_config import (
    PROVIDER_MAP,
    PROVIDER_EMBEDDINGS,
    PROVIDER_EMBEDDING_MODELS,
)


class ShortTermMemory:
    def __init__(
        self,
        trajectory: List[List[ResourceRecord]],
        chat_history: List[dict],
        llm_config: LLMConfig,
    ):
        """_summary_
        Represents the short-term memory of a conversation, storing a trajectory of
        ResourceRecords across multiple turns. This memory enables retrieval of past
        context, intents, tasks, and outputs for use in dynamic and contextual reasoning.


        Args:
            trajectory (List[List[ResourceRecord]]): Memory structure for the conversation where
                                                    each list of ResourceRecord objects encompasses
                                                    the information of a single conversation turn
            chat_history (List[dict]): List of chat history messages
            **kwargs: Additional arguments including:
                - llm_config (LLMConfig, optional): Configuration for LLM and embedding models
        """
        if trajectory is None:
            trajectory = []
        self.trajectory = trajectory[-5:]  # Use the last 5 turns from the trajectory
        if chat_history is None:
            chat_history = []
        self.chat_history = chat_history[-5:]

        # Use provided llm_config or create from MODEL
        self.embedding_model = PROVIDER_EMBEDDINGS.get(
            llm_config.llm_provider, OpenAIEmbeddings
        )(
            **{"model": PROVIDER_EMBEDDING_MODELS[llm_config.llm_provider]}
            if llm_config.llm_provider != "anthropic"
            else {"model_name": PROVIDER_EMBEDDING_MODELS[llm_config.llm_provider]}
        )
        self.llm = PROVIDER_MAP.get(llm_config.llm_provider, ChatOpenAI)(
            model=llm_config.model_type_or_path
        )

    def retrieve_records(
        self, query: str, top_k: int = 3, threshold: float = 0.7
    ) -> Tuple[bool, List[ResourceRecord]]:
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

        query_embedding = np.array(self.embedding_model.embed_query(query)).reshape(
            1, -1
        )
        scored_records = []

        weights = {
            "task": 0.2,
            "intent": 0.35,  # Increased weight for intent
            "context": 0.2,
            "output": 0.2,
            "recency": 0.05,  # Reduced weight for recency
        }

        # Loop through the trajectory and score the records
        for turn_idx, turn in enumerate(self.trajectory):
            recency_score = (turn_idx + 1) / 5
            for record in turn:
                score_components = {
                    "task": 0.0,
                    "intent": 0.0,
                    "context": 0.0,
                    "output": 0.0,
                    "recency": recency_score,
                }

                # Calculate task similarity
            task = record.info.get("attribute", {}).get("task")

            if task:
                task_embedding = np.array(
                    self.embedding_model.embed_query(task)
                ).reshape(1, -1)
                task_similarity = cosine_similarity(query_embedding, task_embedding)[0][
                    0
                ]
                score_components["task"] = task_similarity

            # Calculate intent similarity
            if record.intent:
                intent_embedding = np.array(
                    self.embedding_model.embed_query(record.personalized_intent)
                ).reshape(1, -1)
                intent_similarity = cosine_similarity(
                    query_embedding, intent_embedding
                )[0][0]
                score_components["intent"] = intent_similarity

            # Calculate context similarity
            for step in record.steps or []:
                if isinstance(step, dict) and "context_generate" in step:
                    context_embedding = np.array(
                        self.embedding_model.embed_query(step["context_generate"])
                    ).reshape(1, -1)
                    context_similarity = cosine_similarity(
                        query_embedding, context_embedding
                    )[0][0]
                    score_components["context"] = context_similarity
                    break

            # Calculate output similarity
            if record.output:
                output_embedding = np.array(
                    self.embedding_model.embed_query(record.output)
                ).reshape(1, -1)
                output_similarity = cosine_similarity(
                    query_embedding, output_embedding
                )[0][0]
                score_components["output"] = output_similarity

            # Compute weighted score
            weighted_score = sum(
                score_components[key] * weights[key] for key in score_components
            )
            # Normalize weighted score (optional)
            weighted_score /= sum(weights.values())
            scored_records.append({"record": record, "score": weighted_score})

        # Filter out the records that have a score below the threshold
        relevant_records = [r for r in scored_records if r["score"] >= threshold]
        if not relevant_records:
            return False, []

        # Sort the relevant records by score and return the top_k
        relevant_records.sort(key=lambda x: x["score"], reverse=True)
        return True, [r["record"] for r in relevant_records[:top_k]]

    def retrieve_intent(
        self, query: str, threshold: float = 0.7
    ) -> Tuple[bool, Optional[str]]:
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

        query_embedding = np.array(self.embedding_model.embed_query(query)).reshape(
            1, -1
        )
        best_intent = None
        best_score = -1.0

        # Loop through the trajectory and score the records
        for turn_idx, turn in enumerate(self.trajectory):
            for record in turn:
                if record.intent:
                    intent_embedding = np.array(
                        self.embedding_model.embed_query(record.personalized_intent)
                    ).reshape(1, -1)
                    similarity = cosine_similarity(query_embedding, intent_embedding)[
                        0
                    ][0]
                    if similarity > best_score:
                        best_score = similarity
                        best_intent = record.intent

        # If the best score is above the threshold, return the intent
        if best_score >= threshold:
            return True, best_intent
        else:
            return False, None

    async def personalize(self):
        """
        Loops through the last 5 records in the trajectory and generates personalized intents
        only if not already computed, storing them as `personalized_intent`.
        """
        # Loop through the trajectory and score the records
        tasks = []
        for turn_idx, turn in enumerate(self.trajectory):
            user_utterance = ""
            if (
                turn_idx < len(self.chat_history)
                and self.chat_history[turn_idx].get("role") == "user"
            ):
                user_utterance = self.chat_history[turn_idx]["content"]

            for record in turn:
                # Check if personalized_intent is already set
                if not record.personalized_intent:
                    tasks.append(self._set_personalized_intent(record, user_utterance))

        if tasks:
            await asyncio.gather(*tasks)

    async def _set_personalized_intent(
        self, record: ResourceRecord, user_utterance: str
    ):
        record.personalized_intent = (
            await self.generate_personalized_product_attribute_intent(
                record, user_utterance
            )
        )

    async def generate_personalized_product_attribute_intent(
        self, record: ResourceRecord, user_utterance: str
    ) -> str:
        """
        Args:
            user_utterance (str): User utterance in chat_history corresponding to the record.
            record (ResourceRecord): Record having the information about agent trajectory.

        Returns:
            personalized_intent (str):Generate a more personalized intent using task, tool output, context generate, and intent,
        focusing on product and attribute mentioned or inferred.
        """

        task = record.info.get("attribute", {}).get("task", "") or ""
        tool_output = record.output or ""
        context_generate = ""
        for step in record.steps or []:
            if isinstance(step, dict) and "context_generate" in step:
                context_generate = step["context_generate"]
                break
        user_intent = record.intent or ""

        # Input Section
        inputs_section = f"""
            This is the input information:
            - Tool's final raw output: {tool_output}
            - Task performed by the tool: {task}
            - Tool's context generated response: {context_generate}
            - Basic User Intent: {user_intent}
            - User utterance: {user_utterance}
            """

        prompt = (
            intro.strip()
            + "\nHere are the exemplars.\n"
            + examples.strip()
            + "\n\n"
            + output_instructions.strip()
            + "\n\n"
            + inputs_section.strip()
        )
        response = await self.llm.ainvoke(prompt)
        content = (
            response.get("content")
            if isinstance(response, dict)
            else getattr(response, "content", "")
        )

        match = re.search(
            r"Personalized Intent:\s*(.+)", content, re.IGNORECASE | re.DOTALL
        )
        personalized_intent = match.group(1).strip() if match else content.strip()

        # print(f"Generated personalized product-attribute intent: {personalized_intent}")

        return personalized_intent
