"""Core memory implementation for the Arklex framework.

This module provides the core memory functionality for managing conversation context and history.
The ShortTermMemory class is responsible for storing and retrieving conversation trajectories,
managing embeddings for semantic search, and personalizing user intents based on conversation context.
It includes functionality for retrieving relevant records, managing embeddings with caching,
and generating personalized intents from user interactions.
"""

# TODO(christian): fix annotations in this file.

import asyncio
import re

import numpy as np
from langchain_openai import OpenAIEmbeddings
from Levenshtein import ratio
from sklearn.metrics.pairwise import cosine_similarity

from arklex.memory.entities.memory_entities import ResourceRecord
from arklex.memory.prompts import final_examples, intro, output_instructions
from arklex.types import LLMConfig
from arklex.utils.model_provider_config import (
    PROVIDER_EMBEDDING_MODELS,
    PROVIDER_EMBEDDINGS,
)
from arklex.utils.provider_utils import validate_and_get_model_class


class ShortTermMemory:
    def __init__(
        self,
        trajectory: list[list[ResourceRecord]],
        chat_history: str,
        llm_config: LLMConfig,
    ) -> None:
        """Initialize the ShortTermMemory instance.

        This function initializes the short-term memory with conversation trajectory,
        chat history, and language model configuration. It sets up the embedding model
        and caching mechanism for efficient semantic search.

        Args:
            trajectory (List[List[ResourceRecord]]): Memory structure for the conversation where
                each list of ResourceRecord objects encompasses the information of a single
                conversation turn.
            chat_history (str): Formatted chat history string containing recent conversation turns.
            llm_config (LLMConfig): Configuration for the language model and embeddings.
        """
        if trajectory is None:
            trajectory = []
        self.trajectory = trajectory[-5:]  # Use the last 5 turns from the trajectory

        if chat_history is None:
            chat_history = ""

        # Get last 5 turns from chat history string
        chat_lines = chat_history.split("\n")
        turns = []
        current_turn = []

        for line in chat_lines:
            if line.strip():  # Skip empty lines
                current_turn.append(line)
                if line.startswith("user:"):  # End of a turn
                    turns.append("\n".join(current_turn))
                    current_turn = []

        # Take last 5 turns
        self.chat_history = "\n".join(turns[-5:])

        # Initialize embedding model with caching
        self.embedding_model = PROVIDER_EMBEDDINGS.get(
            llm_config.llm_provider, OpenAIEmbeddings
        )(
            **{"model": PROVIDER_EMBEDDING_MODELS[llm_config.llm_provider]}
            if llm_config.llm_provider != "anthropic"
            else {"model_name": PROVIDER_EMBEDDING_MODELS[llm_config.llm_provider]}
        )
        model_class = validate_and_get_model_class(llm_config)

        self.llm = model_class(model=llm_config.model_type_or_path)

        # Initialize embedding cache
        self._embedding_cache = {}

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text with caching.

        This function retrieves or computes the embedding for a given text, using
        caching to improve performance for repeated queries.

        Args:
            text (str): The text to get the embedding for.

        Returns:
            np.ndarray: The embedding vector for the text.
        """
        if text not in self._embedding_cache:
            self._embedding_cache[text] = np.array(
                self.embedding_model.embed_query(text)
            ).reshape(1, -1)
        return self._embedding_cache[text]

    async def _batch_get_embeddings(self, texts: list[str]) -> list[np.ndarray]:
        """Get embeddings for multiple texts in parallel.

        This function efficiently computes embeddings for multiple texts using
        asynchronous processing.

        Args:
            texts (List[str]): List of texts to get embeddings for.

        Returns:
            List[np.ndarray]: List of embedding vectors for the texts.
        """
        tasks = [asyncio.create_task(self._get_embedding_async(text)) for text in texts]
        return await asyncio.gather(*tasks)

    async def _get_embedding_async(self, text: str) -> np.ndarray:
        """Async wrapper for getting embeddings.

        This function provides an asynchronous interface for getting text embeddings.

        Args:
            text (str): The text to get the embedding for.

        Returns:
            np.ndarray: The embedding vector for the text.
        """
        return self._get_embedding(text)

    def retrieve_records(
        self,
        query: str,
        top_k: int = 3,
        threshold: float = 0.55,
        cosine_threshold: float = 0.7,
    ) -> tuple[bool, list[ResourceRecord]]:
        """Retrieve relevant records from memory based on a query.

        This function searches through the conversation trajectory to find records
        that are semantically similar to the query. It uses a combination of
        cosine similarity and string matching to find the most relevant records.

        Args:
            query (str): The query string to retrieve relevant records for.
            top_k (int, optional): The number of top records to return. Defaults to 3.
            threshold (float, optional): The string similarity score threshold for filtering
                relevant records. Defaults to 0.55.
            cosine_threshold (float, optional): The cosine similarity threshold for initial
                filtering. Defaults to 0.7.

        Returns:
            Tuple[bool, List[ResourceRecord]]: A tuple where the first element is a boolean
                indicating whether relevant records were found, and the second element is a
                list of the top-k relevant ResourceRecord objects based on the query.
        """
        if not self.trajectory:
            return False, []

        query_embedding = self._get_embedding(query)
        scored_records = []

        weights = {
            "task": 0.1,
            "intent": 0.50,  # Increased weight for intent
            "context": 0.15,
            "output": 0.2,
            "recency": 0.05,  # Reduced weight for recency
        }

        # Loop through the trajectory and score the records
        for _turn_idx, turn in enumerate(self.trajectory):
            if not turn:  # Skip empty turns
                continue

            recency_score = (_turn_idx + 1) / 5
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
                    task_embedding = self._get_embedding(task)
                    task_similarity = cosine_similarity(
                        query_embedding, task_embedding
                    )[0][0]
                    score_components["task"] = task_similarity

                # Calculate intent similarity
                if record.personalized_intent:
                    match = re.search(
                        r"intent:\s*(.+?)\s*product:\s*(.+?)\s*attribute:\s*(.+)",
                        record.personalized_intent,
                        re.IGNORECASE,
                    )
                    if match:
                        personalized_intent = match.group(1).strip().lower()
                        product = match.group(2).strip().lower()
                        attribute = match.group(3).strip().lower()

                        # First check cosine similarity of personalized intent as a filter
                        intent_embedding = self._get_embedding(personalized_intent)
                        cosine_similarity_score = cosine_similarity(
                            query_embedding, intent_embedding
                        )[0][0]

                        # Only proceed with string comparison if cosine similarity is above threshold
                        if cosine_similarity_score > cosine_threshold:
                            # Combine product and attribute for string comparison
                            combined = f"{attribute} {product}"
                            string_similarity = ratio(query, combined)
                            score_components["intent"] = string_similarity
                        else:
                            score_components["intent"] = 0.0
                    else:
                        score_components["intent"] = 0.0
                else:
                    score_components["intent"] = 0.0

                # Calculate context similarity
                for step in record.steps or []:
                    if isinstance(step, dict) and "context_generate" in step:
                        context_embedding = self._get_embedding(
                            step["context_generate"]
                        )
                        context_similarity = cosine_similarity(
                            query_embedding, context_embedding
                        )[0][0]
                        score_components["context"] = context_similarity
                        break

                # Calculate output similarity
                if record.output:
                    output_embedding = self._get_embedding(record.output)
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
        self, query: str, string_threshold: float = 0.4, cosine_threshold: float = 0.7
    ) -> tuple[bool, str | None]:
        """Retrieve the most relevant intent from memory based on a query.

        This function searches through the conversation trajectory to find the intent
        that best matches the query. It uses a combination of cosine similarity and
        string matching to find the most relevant intent.

        Args:
            query (str): The query string to retrieve the most relevant intent for.
            string_threshold (float, optional): The string similarity score threshold for
                filtering relevant intents. Defaults to 0.4.
            cosine_threshold (float, optional): The cosine similarity threshold for initial
                filtering. Defaults to 0.7.

        Returns:
            Tuple[bool, Optional[str]]: A tuple where the first element is a boolean
                indicating whether a relevant intent was found, and the second element is
                the most relevant intent (if found), or None if no relevant intent meets
                the threshold.
        """
        if not self.trajectory:
            return False, None

        query_embedding = self._get_embedding(query)
        best_score = -1
        best_intent = None

        # Loop through the trajectory and score the records
        for _turn_idx, turn in enumerate(self.trajectory):
            for record in turn:
                if record.personalized_intent:
                    match = re.search(
                        r"intent:\s*(.+?)\s*product:\s*(.+?)\s*attribute:\s*(.+)",
                        record.personalized_intent,
                        re.IGNORECASE,
                    )
                    if match:
                        personalized_intent = match.group(1).strip().lower()
                        product = match.group(2).strip().lower()
                        attribute = match.group(3).strip().lower()

                        # First check cosine similarity of personalized intent as a filter
                        intent_embedding = self._get_embedding(personalized_intent)
                        cosine_similarity_score = cosine_similarity(
                            query_embedding, intent_embedding
                        )[0][0]

                        # Only proceed with string comparison if cosine similarity is above threshold
                        if cosine_similarity_score > cosine_threshold:
                            # Combine product and attribute for string comparison
                            combined = f"{attribute} {product}"
                            string_similarity = ratio(query, combined)

                            # Use string similarity as the final score
                            if string_similarity > best_score:
                                best_score = string_similarity
                                best_intent = record.intent

        # If the best score is above the threshold, return the intent
        if best_score >= string_threshold:
            return True, best_intent
        else:
            return False, None

    async def personalize(self) -> None:
        """Generate personalized intents for records in memory.

        This function processes the conversation trajectory to generate personalized
        intents for each record. It uses the language model to create more specific
        and contextual intents based on the conversation history.
        """
        # Get all user utterances from chat history
        user_utterances = []
        for line in self.chat_history.split("\n"):
            if line.startswith("user:"):
                user_utterances.append(line.replace("user:", "").strip())

        # Loop through the trajectory and score the records
        tasks = []
        for _turn_idx, turn in enumerate(self.trajectory):
            # Get corresponding user utterance for this turn
            user_utterance = (
                user_utterances[_turn_idx] if _turn_idx < len(user_utterances) else ""
            )
            for record in turn:
                # Check if personalized_intent is already set
                if not record.personalized_intent:
                    tasks.append(self._set_personalized_intent(record, user_utterance))

        if tasks:
            await asyncio.gather(*tasks)

    async def _set_personalized_intent(
        self, record: ResourceRecord, user_utterance: str
    ) -> None:
        """Set personalized intent for a record.

        This function generates and sets a personalized intent for a given record
        based on the user's utterance.

        Args:
            record (ResourceRecord): The record to set the personalized intent for.
            user_utterance (str): The user's utterance to consider for personalization.
        """
        record.personalized_intent = (
            await self.generate_personalized_product_attribute_intent(
                record, user_utterance
            )
        )

    async def generate_personalized_product_attribute_intent(
        self, record: ResourceRecord, user_utterance: str
    ) -> str:
        """Generate a personalized intent focusing on product and attribute.

        This function creates a personalized intent that emphasizes the product and
        attribute aspects of the interaction, using the record's information and
        user utterance.

        Args:
            record (ResourceRecord): The record containing information about the interaction.
            user_utterance (str): The user's utterance to consider for personalization.

        Returns:
            str: The generated personalized intent string.
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
            + final_examples.strip()
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

        return personalized_intent
