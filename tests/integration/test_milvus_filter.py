"""
Integration tests for Milvus filter taskgraph.

This module contains comprehensive integration tests for the Milvus filter taskgraph,
including proper mocking of external services, RAG functionality with product filtering,
node transitions, and edge case testing. These tests validate the complete Milvus
integration workflow from user input to filtered response generation.
"""

from typing import Any
from unittest.mock import Mock, patch

from arklex.orchestrator.entities.msg_state_entities import MessageState
from tests.integration.integration_test_utils import MilvusTestHelper


class TestMilvusFilterIntegration:
    """
    Integration tests for Milvus filter taskgraph.

    This test class validates the complete Milvus integration workflow,
    including taskgraph structure, worker configuration, RAG functionality,
    product filtering, and error handling scenarios.
    """

    def test_taskgraph_structure_validation(
        self, load_milvus_config: dict[str, Any]
    ) -> None:
        """
        Test that the taskgraph has the correct structure and required fields.

        This test validates that the Milvus taskgraph configuration contains
        all necessary components for proper functionality, including nodes,
        edges, model configuration, and worker setup.
        """
        MilvusTestHelper.validate_taskgraph_structure(load_milvus_config)

    def test_worker_configuration(self, load_milvus_config: dict[str, Any]) -> None:
        """
        Test that workers are properly configured in the taskgraph.

        This test ensures that the MilvusRAGWorker and other required workers
        are properly configured with the correct parameters and settings.
        """
        MilvusTestHelper.validate_worker_configuration(load_milvus_config)

    def test_domain_specific_configuration(
        self, load_milvus_config: dict[str, Any]
    ) -> None:
        """
        Test that the taskgraph is properly configured for robotics domain.

        This test validates that the taskgraph is specifically configured
        for robotics domain queries and contains appropriate filtering logic.
        """
        MilvusTestHelper.validate_domain_specific_configuration(load_milvus_config)

    @patch("arklex.env.env.Environment.step")
    @patch("arklex.orchestrator.orchestrator.post_process_response")
    @patch("langchain_openai.embeddings.base.OpenAIEmbeddings.embed_documents")
    @patch("arklex.utils.model_provider_config.PROVIDER_MAP")
    @patch(
        "arklex.env.tools.RAG.retrievers.milvus_retriever.RetrieveEngine.milvus_retrieve"
    )
    @patch("arklex.env.tools.utils.ToolGenerator.context_generate")
    @patch("arklex.orchestrator.NLU.core.intent.IntentDetector.execute")
    @patch("arklex.env.planner.react_planner.FAISS")
    @patch("arklex.env.tools.RAG.retrievers.faiss_retriever.FAISS")
    def test_start_message_delivery(
        self,
        mock_faiss_retriever: Mock,
        mock_faiss_planner: Mock,
        mock_intent_detector: Mock,
        mock_generator: Mock,
        mock_milvus_retrieve: Mock,
        mock_provider_map: Mock,
        mock_embeddings: Mock,
        mock_post_process: Mock,
        mock_env_step: Mock,
        config_and_env: tuple[dict[str, Any], Any, str],
    ) -> None:
        """
        Test that the start message is delivered correctly.

        This test validates that the system properly handles the initial
        start message and delivers the appropriate welcome response to users.
        """
        config, env, start_message = config_and_env

        # Mock intent detection to stay at start node
        # This simulates the system recognizing that we should remain at the start
        mock_intent_detector.return_value = "0) Stay at start"

        # Mock the provider map to return a mock ChatOpenAI
        # This simulates the LLM provider configuration for response generation
        mock_chat = Mock()
        mock_response = Mock()
        mock_response.content = start_message
        mock_chat.return_value = mock_response
        mock_provider_map.__getitem__.return_value = mock_chat

        # Mock embeddings to return consistent vectors
        # This simulates the embedding generation for similarity search
        mock_embeddings.return_value = [[0.1] * 1536] * 3

        # Mock FAISS operations for both planner and retriever
        # This simulates the vector database operations
        mock_vectorstore = Mock()
        mock_vectorstore.similarity_search_with_score.return_value = [
            (Mock(metadata={"resource_name": "test_resource"}), 0.8)
        ]
        mock_faiss_planner.return_value = mock_vectorstore
        mock_faiss_retriever.return_value = mock_vectorstore

        # Mock ToolGenerator to return proper MessageState
        # This simulates the context generation process
        def mock_context_generate_side_effect(
            message_state: MessageState,
        ) -> MessageState:
            message_state.response = start_message
            return message_state

        mock_generator.side_effect = mock_context_generate_side_effect

        # Mock post-processing to return the same message_state
        # This simulates the final response processing step
        def mock_post_process_side_effect(
            message_state: MessageState | None,
            params: dict[str, Any],
            hitl_available: bool,
            hitl_enabled: bool,
        ) -> MessageState:
            if message_state is None:
                message_state = MilvusTestHelper.create_mock_message_state(
                    start_message
                )
            return message_state

        mock_post_process.side_effect = mock_post_process_side_effect

        # Mock environment step to return the start message
        # This simulates the environment execution step
        def mock_env_step_side_effect(
            resource_id: str,
            message_state: MessageState,
            params: dict[str, Any],
            node_info: dict[str, Any],
        ) -> tuple[MessageState, dict[str, Any]]:
            # Set the response to the start message
            message_state.response = start_message
            return message_state, params

        mock_env_step.side_effect = mock_env_step_side_effect

        # Test start message delivery with empty conversation context
        history: list[dict[str, str]] = []
        params: dict[str, Any] = {}
        user_text = "<start>"

        output, params, hitl = MilvusTestHelper.get_api_bot_response(
            config, env, user_text, history, params
        )

        # Verify start message is delivered correctly
        assert start_message in output, "Start message not delivered correctly"
        assert hitl is None or hitl == "", "HITL should not be active for start message"

    @patch("arklex.orchestrator.orchestrator.post_process_response")
    @patch("langchain_openai.embeddings.base.OpenAIEmbeddings.embed_documents")
    @patch("arklex.utils.model_provider_config.PROVIDER_MAP")
    @patch(
        "arklex.env.tools.RAG.retrievers.milvus_retriever.RetrieveEngine.milvus_retrieve"
    )
    @patch("arklex.env.tools.utils.ToolGenerator.context_generate")
    @patch("arklex.orchestrator.NLU.core.intent.IntentDetector.execute")
    @patch("arklex.env.planner.react_planner.FAISS")
    @patch("arklex.env.tools.RAG.retrievers.faiss_retriever.FAISS")
    def test_milvus_rag_worker_product_query(
        self,
        mock_faiss_retriever: Mock,
        mock_faiss_planner: Mock,
        mock_intent_detector: Mock,
        mock_generator: Mock,
        mock_milvus_retrieve: Mock,
        mock_provider_map: Mock,
        mock_embeddings: Mock,
        mock_post_process: Mock,
        config_and_env: tuple[dict[str, Any], Any, str],
    ) -> None:
        """
        Test MilvusRAGWorker responding to product questions with filtering.

        This test validates that the MilvusRAGWorker properly handles
        product queries, applies filtering tags, and generates appropriate
        responses based on retrieved information.
        """
        config, env, start_message = config_and_env

        # Mock intent detection to transition to MilvusRAGWorker
        # This simulates the system recognizing a product query
        mock_intent_detector.return_value = "1) User seeks information about robots"

        # Mock Milvus retrieval with product filtering
        # This simulates the filtered retrieval process with proper tag validation
        def mock_milvus_retrieve_side_effect(
            message_state: MessageState, tags: dict[str, str] | None = None
        ) -> MessageState:
            # Verify that product tags are passed correctly for filtering
            assert tags is not None, "Tags should be passed to Milvus retrieval"
            assert "product" in tags, "Product tag should be present"
            assert tags["product"] == "robots", "Product tag should be 'robots'"

            # Add retrieved context to message state
            # This simulates the information retrieval process
            message_state.message_flow = (
                "Retrieved information about ADAM robot bartender"
            )
            return message_state

        mock_milvus_retrieve.side_effect = mock_milvus_retrieve_side_effect

        # Mock ToolGenerator to generate response based on retrieved context
        # This simulates the response generation process
        def mock_context_generate_side_effect(
            message_state: MessageState,
        ) -> MessageState:
            message_state.response = (
                "Based on the retrieved information, the ADAM robot is a bartender "
                "that makes tea, coffee, and cocktails. It's available for both "
                "purchase and rental for multiple purposes."
            )
            return message_state

        mock_generator.side_effect = mock_context_generate_side_effect

        # Mock the provider map to return a mock ChatOpenAI
        # This simulates the LLM provider for response generation
        mock_chat = Mock()
        mock_response = Mock()
        mock_response.content = "Product information response"
        mock_chat.return_value = mock_response
        mock_provider_map.__getitem__.return_value = mock_chat

        # Mock embeddings for similarity search
        # This simulates the embedding generation process
        mock_embeddings.return_value = [[0.1] * 1536] * 3

        # Mock FAISS operations for vector database
        # This simulates the vector similarity search
        mock_vectorstore = Mock()
        mock_vectorstore.similarity_search_with_score.return_value = [
            (Mock(metadata={"resource_name": "test_resource"}), 0.8)
        ]
        mock_faiss_planner.return_value = mock_vectorstore
        mock_faiss_retriever.return_value = mock_vectorstore

        # Mock post-processing to handle the final response
        # This simulates the final response processing step
        def mock_post_process_side_effect(
            message_state: MessageState | None,
            params: dict[str, Any],
            hitl_available: bool,
            hitl_enabled: bool,
        ) -> MessageState:
            if message_state is None:
                message_state = MilvusTestHelper.create_mock_message_state(
                    "Product information response"
                )
            return message_state

        mock_post_process.side_effect = mock_post_process_side_effect

        # Test product query with conversation context
        history: list[dict[str, str]] = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "How can I help you?"},
        ]
        params: dict[str, Any] = {}
        user_text = "Tell me about your robots"

        output, params, hitl = MilvusTestHelper.get_api_bot_response(
            config, env, user_text, history, params
        )

        # Verify that the response contains product information
        assert "ADAM robot" in output, "Response should contain product information"
        assert "bartender" in output, "Response should mention robot capabilities"
        assert hitl is None or hitl == "", (
            "HITL should not be active for product queries"
        )

    @patch("arklex.orchestrator.orchestrator.post_process_response")
    @patch("langchain_openai.embeddings.base.OpenAIEmbeddings.embed_documents")
    @patch("langchain_openai.chat_models.base.ChatOpenAI")
    @patch(
        "arklex.env.tools.RAG.retrievers.milvus_retriever.RetrieveEngine.milvus_retrieve"
    )
    @patch("arklex.env.tools.utils.ToolGenerator.context_generate")
    @patch("arklex.orchestrator.NLU.core.intent.IntentDetector.execute")
    @patch("arklex.env.planner.react_planner.FAISS")
    @patch("arklex.env.tools.RAG.retrievers.faiss_retriever.FAISS")
    def test_conversation_flow_multiple_turns(
        self,
        mock_faiss_retriever: Mock,
        mock_faiss_planner: Mock,
        mock_intent_detector: Mock,
        mock_generator: Mock,
        mock_milvus_retrieve: Mock,
        mock_provider_map: Mock,
        mock_embeddings: Mock,
        mock_post_process: Mock,
        config_and_env: tuple[dict[str, Any], Any, str],
    ) -> None:
        """
        Test conversation flow with multiple turns and context persistence.

        This test validates that the system maintains conversation context
        across multiple turns and properly handles follow-up questions
        with appropriate filtering and response generation.
        """
        config, env, start_message = config_and_env

        # Mock intent detection for first turn (product query)
        mock_intent_detector.return_value = "1) User seeks information about robots"

        # Mock Milvus retrieval for first turn
        # This simulates the initial product information retrieval
        def mock_milvus_retrieve_side_effect(
            message_state: MessageState, tags: dict[str, str] | None = None
        ) -> MessageState:
            # Verify filtering tags are applied correctly
            assert tags is not None, "Tags should be passed to Milvus retrieval"
            assert "product" in tags, "Product tag should be present"

            # Add retrieved context for first turn
            message_state.message_flow = (
                "Retrieved information about ADAM robot bartender capabilities"
            )
            return message_state

        mock_milvus_retrieve.side_effect = mock_milvus_retrieve_side_effect

        # Mock response generation for first turn
        # This simulates the initial response generation
        def mock_context_generate_side_effect(
            message_state: MessageState,
        ) -> MessageState:
            message_state.response = (
                "The ADAM robot is a bartender that can make various beverages. "
                "What specific information would you like to know about it?"
            )
            return message_state

        mock_generator.side_effect = mock_context_generate_side_effect

        # Mock embeddings and other dependencies
        mock_embeddings.return_value = [[0.1] * 1536] * 3
        mock_vectorstore = Mock()
        mock_vectorstore.similarity_search_with_score.return_value = [
            (Mock(metadata={"resource_name": "test_resource"}), 0.8)
        ]
        mock_faiss_planner.return_value = mock_vectorstore
        mock_faiss_retriever.return_value = mock_vectorstore

        # Mock post-processing for first turn
        def mock_post_process_side_effect(
            message_state: MessageState | None,
            params: dict[str, Any],
            hitl_available: bool,
            hitl_enabled: bool,
        ) -> MessageState:
            if message_state is None:
                message_state = MilvusTestHelper.create_mock_message_state(
                    "The ADAM robot is a bartender that can make various beverages."
                )
            return message_state

        mock_post_process.side_effect = mock_post_process_side_effect

        # Test first turn - initial product query
        history: list[dict[str, str]] = []
        params: dict[str, Any] = {}
        user_text = "Tell me about your robots"

        output1, params, hitl = MilvusTestHelper.get_api_bot_response(
            config, env, user_text, history, params
        )

        # Verify first response
        assert "ADAM robot" in output1, "First response should mention ADAM robot"
        assert "bartender" in output1, (
            "First response should mention bartender capabilities"
        )

        # Mock intent detection for second turn (follow-up question)
        mock_intent_detector.return_value = "1) User seeks detailed product information"

        # Mock Milvus retrieval for second turn with different context
        # This simulates follow-up information retrieval
        def mock_milvus_retrieve_side_effect_second(
            message_state: MessageState, tags: dict[str, str] | None = None
        ) -> MessageState:
            # Verify filtering tags are still applied correctly
            assert tags is not None, "Tags should be passed to Milvus retrieval"
            assert "product" in tags, "Product tag should be present"

            # Add retrieved context for second turn
            message_state.message_flow = (
                "Retrieved detailed specifications and pricing for ADAM robot"
            )
            return message_state

        mock_milvus_retrieve.side_effect = mock_milvus_retrieve_side_effect_second

        # Mock response generation for second turn
        # This simulates the follow-up response generation
        def mock_context_generate_side_effect_second(
            message_state: MessageState,
        ) -> MessageState:
            message_state.response = (
                "The ADAM robot can make tea, coffee, and cocktails. "
                "It features advanced automation and costs $50,000. "
                "Would you like to know about delivery times or other models?"
            )
            return message_state

        mock_generator.side_effect = mock_context_generate_side_effect_second

        # Mock post-processing for second turn
        def mock_post_process_side_effect_second(
            message_state: MessageState | None,
            params: dict[str, Any],
            hitl_available: bool,
            hitl_enabled: bool,
        ) -> MessageState:
            if message_state is None:
                message_state = MilvusTestHelper.create_mock_message_state(
                    "The ADAM robot can make tea, coffee, and cocktails."
                )
            return message_state

        mock_post_process.side_effect = mock_post_process_side_effect_second

        # Test second turn - follow-up question
        history = [
            {"role": "user", "content": "Tell me about your robots"},
            {"role": "assistant", "content": output1},
        ]
        user_text = "What can it make and how much does it cost?"

        output2, params, hitl = MilvusTestHelper.get_api_bot_response(
            config, env, user_text, history, params
        )

        # Verify second response contains detailed information
        assert "tea" in output2, "Second response should mention tea"
        assert "coffee" in output2, "Second response should mention coffee"
        assert "cocktails" in output2, "Second response should mention cocktails"
        assert "$50,000" in output2, "Second response should mention pricing"
        assert hitl is None or hitl == "", (
            "HITL should not be active for follow-up queries"
        )

    @patch("arklex.orchestrator.orchestrator.post_process_response")
    @patch("langchain_openai.embeddings.base.OpenAIEmbeddings.embed_documents")
    @patch("langchain_openai.chat_models.base.ChatOpenAI")
    @patch(
        "arklex.env.tools.RAG.retrievers.milvus_retriever.RetrieveEngine.milvus_retrieve"
    )
    @patch("arklex.env.tools.utils.ToolGenerator.context_generate")
    @patch("arklex.orchestrator.NLU.core.intent.IntentDetector.execute")
    @patch("arklex.env.planner.react_planner.FAISS")
    @patch("arklex.env.tools.RAG.retrievers.faiss_retriever.FAISS")
    def test_edge_case_empty_input(
        self,
        mock_faiss_retriever: Mock,
        mock_faiss_planner: Mock,
        mock_intent_detector: Mock,
        mock_generator: Mock,
        mock_milvus_retrieve: Mock,
        mock_provider_map: Mock,
        mock_embeddings: Mock,
        mock_post_process: Mock,
        config_and_env: tuple[dict[str, Any], Any, str],
    ) -> None:
        """
        Test edge case handling for empty or invalid input.

        This test validates that the system properly handles edge cases
        such as empty input, whitespace-only input, and other invalid
        user inputs without crashing or producing unexpected behavior.
        """
        config, env, start_message = config_and_env

        # Mock intent detection to handle empty input gracefully
        # This simulates the system's response to invalid input
        mock_intent_detector.return_value = "0) Stay at start"

        # Mock response generation for empty input
        # This simulates the system's graceful handling of edge cases
        def mock_context_generate_side_effect(
            message_state: MessageState,
        ) -> MessageState:
            message_state.response = (
                "I didn't quite catch that. Could you please rephrase your question?"
            )
            return message_state

        mock_generator.side_effect = mock_context_generate_side_effect

        # Mock embeddings and other dependencies
        mock_embeddings.return_value = [[0.1] * 1536] * 3
        mock_vectorstore = Mock()
        mock_vectorstore.similarity_search_with_score.return_value = [
            (Mock(metadata={"resource_name": "test_resource"}), 0.8)
        ]
        mock_faiss_planner.return_value = mock_vectorstore
        mock_faiss_retriever.return_value = mock_vectorstore

        # Mock post-processing for edge case handling
        def mock_post_process_side_effect(
            message_state: MessageState | None,
            params: dict[str, Any],
            hitl_available: bool,
            hitl_enabled: bool,
        ) -> MessageState:
            if message_state is None:
                message_state = MilvusTestHelper.create_mock_message_state(
                    "I didn't quite catch that. Could you please rephrase your question?"
                )
            return message_state

        mock_post_process.side_effect = mock_post_process_side_effect

        # Test various edge cases
        edge_cases = ["", "   ", "\n", "\t", "   \n   "]

        for empty_input in edge_cases:
            history: list[dict[str, str]] = []
            params: dict[str, Any] = {}

            output, params, hitl = MilvusTestHelper.get_api_bot_response(
                config, env, empty_input, history, params
            )

            # Verify that the system handles empty input gracefully
            assert "rephrase" in output.lower() or "didn't catch" in output.lower(), (
                f"System should handle empty input gracefully, got: {output}"
            )
            assert hitl is None or hitl == "", (
                "HITL should not be active for empty input"
            )

    @patch("arklex.orchestrator.orchestrator.post_process_response")
    @patch("langchain_openai.embeddings.base.OpenAIEmbeddings.embed_documents")
    @patch("langchain_openai.chat_models.base.ChatOpenAI")
    @patch(
        "arklex.env.tools.RAG.retrievers.milvus_retriever.RetrieveEngine.milvus_retrieve"
    )
    @patch("arklex.env.tools.utils.ToolGenerator.context_generate")
    @patch("arklex.orchestrator.NLU.core.intent.IntentDetector.execute")
    @patch("arklex.env.planner.react_planner.FAISS")
    @patch("arklex.env.tools.RAG.retrievers.faiss_retriever.FAISS")
    def test_milvus_retrieval_error_handling(
        self,
        mock_faiss_retriever: Mock,
        mock_faiss_planner: Mock,
        mock_intent_detector: Mock,
        mock_generator: Mock,
        mock_milvus_retrieve: Mock,
        mock_provider_map: Mock,
        mock_embeddings: Mock,
        mock_post_process: Mock,
        config_and_env: tuple[dict[str, Any], Any, str],
    ) -> None:
        """
        Test error handling when Milvus retrieval fails.

        This test validates that the system properly handles Milvus
        connection errors, retrieval failures, and other exceptions
        without crashing and provides appropriate error responses.
        """
        config, env, start_message = config_and_env

        # Mock intent detection for product query
        mock_intent_detector.return_value = "1) User seeks information about robots"

        # Mock Milvus retrieval to simulate an error
        # This simulates a connection failure or retrieval error
        def mock_milvus_retrieve_error_side_effect(
            message_state: MessageState, tags: dict[str, str] | None = None
        ) -> MessageState:
            # Simulate retrieval error
            message_state.message_flow = (
                "Error: Failed to retrieve information from Milvus"
            )
            message_state.response = (
                "I'm sorry, I'm having trouble accessing the product database right now. "
                "Please try again in a moment."
            )
            return message_state

        mock_milvus_retrieve.side_effect = mock_milvus_retrieve_error_side_effect

        # Mock response generation for error handling
        # This simulates the system's response to retrieval errors
        def mock_context_generate_side_effect(
            message_state: MessageState,
        ) -> MessageState:
            message_state.response = (
                "I'm sorry, I'm having trouble accessing the product database right now. "
                "Please try again in a moment."
            )
            return message_state

        mock_generator.side_effect = mock_context_generate_side_effect

        # Mock embeddings and other dependencies
        mock_embeddings.return_value = [[0.1] * 1536] * 3
        mock_vectorstore = Mock()
        mock_vectorstore.similarity_search_with_score.return_value = [
            (Mock(metadata={"resource_name": "test_resource"}), 0.8)
        ]
        mock_faiss_planner.return_value = mock_vectorstore
        mock_faiss_retriever.return_value = mock_vectorstore

        # Mock post-processing for error handling
        def mock_post_process_side_effect(
            message_state: MessageState | None,
            params: dict[str, Any],
            hitl_available: bool,
            hitl_enabled: bool,
        ) -> MessageState:
            if message_state is None:
                message_state = MilvusTestHelper.create_mock_message_state(
                    "I'm sorry, I'm having trouble accessing the product database right now."
                )
            return message_state

        mock_post_process.side_effect = mock_post_process_side_effect

        # Test error handling with product query
        history: list[dict[str, str]] = []
        params: dict[str, Any] = {}
        user_text = "Tell me about your robots"

        output, params, hitl = MilvusTestHelper.get_api_bot_response(
            config, env, user_text, history, params
        )

        # Verify that the system handles errors gracefully
        assert "sorry" in output.lower(), "Response should apologize for the error"
        assert "trouble" in output.lower(), "Response should mention the issue"
        assert "try again" in output.lower(), "Response should suggest retrying"
        assert hitl is None or hitl == "", (
            "HITL should not be active for error responses"
        )

    @patch("arklex.orchestrator.orchestrator.post_process_response")
    @patch("langchain_openai.embeddings.base.OpenAIEmbeddings.embed_documents")
    @patch("langchain_openai.chat_models.base.ChatOpenAI")
    @patch(
        "arklex.env.tools.RAG.retrievers.milvus_retriever.RetrieveEngine.milvus_retrieve"
    )
    @patch("arklex.env.tools.utils.ToolGenerator.context_generate")
    @patch("arklex.orchestrator.NLU.core.intent.IntentDetector.execute")
    @patch("arklex.env.planner.react_planner.FAISS")
    @patch("arklex.env.tools.RAG.retrievers.faiss_retriever.FAISS")
    def test_product_filtering_accuracy(
        self,
        mock_faiss_retriever: Mock,
        mock_faiss_planner: Mock,
        mock_intent_detector: Mock,
        mock_generator: Mock,
        mock_milvus_retrieve: Mock,
        mock_chat: Mock,
        mock_embeddings: Mock,
        mock_post_process: Mock,
        config_and_env: tuple[dict[str, Any], Any, str],
    ) -> None:
        """
        Test the accuracy of product filtering functionality.

        This test validates that the system correctly applies product
        filtering tags and retrieves relevant information based on
        the specific product category being queried.
        """
        config, env, start_message = config_and_env

        # Track the tags used for filtering to verify accuracy
        captured_tags = []

        # Mock intent detection for product query
        mock_intent_detector.return_value = "1) User seeks information about robots"

        # Mock Milvus retrieval with tag capture
        # This simulates the filtering process and captures the tags used
        def mock_milvus_retrieve_side_effect(
            message_state: MessageState, tags: dict[str, str] | None = None
        ) -> MessageState:
            # Record the tags used for filtering
            captured_tags.append(tags)

            # Verify that product tags are passed correctly
            assert tags is not None, "Tags should be passed to Milvus retrieval"
            assert "product" in tags, "Product tag should be present"
            assert tags["product"] == "robots", "Product tag should be 'robots'"

            # Add retrieved context based on filtering
            message_state.message_flow = (
                "Retrieved filtered information about robotics products"
            )
            return message_state

        mock_milvus_retrieve.side_effect = mock_milvus_retrieve_side_effect

        # Mock response generation with filtered content
        # This simulates the response generation based on filtered results
        def mock_context_generate_side_effect(
            message_state: MessageState,
        ) -> MessageState:
            message_state.response = (
                "Based on the filtered search results, I found information about "
                "our robotics products. The ADAM robot is our flagship bartender "
                "model, and we also have the ARM robot for different applications."
            )
            return message_state

        mock_generator.side_effect = mock_context_generate_side_effect

        # Mock embeddings and other dependencies
        mock_embeddings.return_value = [[0.1] * 1536] * 3
        mock_vectorstore = Mock()
        mock_vectorstore.similarity_search_with_score.return_value = [
            (Mock(metadata={"resource_name": "test_resource"}), 0.8)
        ]
        mock_faiss_planner.return_value = mock_vectorstore
        mock_faiss_retriever.return_value = mock_vectorstore

        # Mock post-processing for filtered response
        def mock_post_process_side_effect(
            message_state: MessageState | None,
            params: dict[str, Any],
            hitl_available: bool,
            hitl_enabled: bool,
        ) -> MessageState:
            if message_state is None:
                message_state = MilvusTestHelper.create_mock_message_state(
                    "Based on the filtered search results, I found information about robotics products."
                )
            return message_state

        mock_post_process.side_effect = mock_post_process_side_effect

        # Test product filtering with specific query
        history: list[dict[str, str]] = []
        params: dict[str, Any] = {}
        user_text = "What robotics products do you have?"

        output, params, hitl = MilvusTestHelper.get_api_bot_response(
            config, env, user_text, history, params
        )

        # Verify that filtering was applied correctly
        assert len(captured_tags) > 0, "Tags should have been captured"
        assert captured_tags[0] is not None, "Captured tags should not be None"
        assert "product" in captured_tags[0], "Captured tags should contain product key"
        assert captured_tags[0]["product"] == "robots", "Product tag should be 'robots'"

        # Verify that the response contains filtered content
        assert "robotics products" in output, (
            "Response should mention robotics products"
        )
        assert "ADAM robot" in output, "Response should mention ADAM robot"
        assert "ARM robot" in output, "Response should mention ARM robot"
        assert hitl is None or hitl == "", (
            "HITL should not be active for filtered queries"
        )

    def test_environment_initialization(
        self, load_milvus_config: dict[str, Any]
    ) -> None:
        """
        Test that the environment initializes correctly with Milvus configuration.

        This test validates that the environment can be properly initialized
        with the Milvus taskgraph configuration and all required components
        are loaded correctly.
        """
        from arklex.env.env import Environment
        from arklex.orchestrator.NLU.services.model_service import ModelService

        config = load_milvus_config

        # Test environment initialization
        # This validates that the environment can be created with the config
        model_service = ModelService(config["model"])

        env = Environment(
            tools=config.get("tools", []),
            workers=config.get("workers", []),
            agents=config.get("agents", []),
            slot_fill_api=config["slotfillapi"],
            planner_enabled=True,
            model_service=model_service,
        )

        # Verify that the environment was created successfully
        assert env is not None, "Environment should be created successfully"
        assert hasattr(env, "workers"), "Environment should have workers"
        assert hasattr(env, "tools"), "Environment should have tools"

    def test_taskgraph_metadata_validation(
        self, load_milvus_config: dict[str, Any]
    ) -> None:
        """
        Test validation of taskgraph metadata and version information.

        This test validates that the taskgraph contains proper metadata
        and version information for the Milvus integration.
        """
        MilvusTestHelper.validate_taskgraph_metadata(load_milvus_config)

    def test_node_edge_consistency(self, load_milvus_config: dict[str, Any]) -> None:
        """
        Test that nodes and edges are consistent and properly connected.

        This test validates that all edges in the taskgraph reference
        valid nodes and that the graph structure is consistent.
        """
        MilvusTestHelper.validate_node_edge_consistency(load_milvus_config)
