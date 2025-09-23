"""Tests for extract_conversation_info module."""

import pytest
import networkx as nx
from arklex.evaluation.extract_conversation_info import (
    get_edges_and_counts,
    build_intent_graph,
)


class TestGetEdgesAndCounts:
    """Test cases for get_edges_and_counts function."""

    def test_get_edges_and_counts_empty_data(self):
        """Test with empty data list."""
        data = []
        result = get_edges_and_counts(data)
        assert result == {}

    def test_get_edges_and_counts_single_conversation(self):
        """Test with single conversation."""
        data = [
            {
                "conversation": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"},
                    {"role": "user", "content": "How are you?"},
                ]
            }
        ]
        result = get_edges_and_counts(data)
        # Should have edges between consecutive messages
        assert len(result) == 2
        assert ("Hello", "Hi there") in result
        assert ("Hi there", "How are you?") in result
        assert result[("Hello", "Hi there")] == 1
        assert result[("Hi there", "How are you?")] == 1

    def test_get_edges_and_counts_multiple_conversations(self):
        """Test with multiple conversations."""
        data = [
            {
                "conversation": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                ]
            },
            {
                "conversation": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                ]
            },
        ]
        result = get_edges_and_counts(data)
        # Should have same edge with count 2
        assert len(result) == 1
        assert ("Hello", "Hi") in result
        assert result[("Hello", "Hi")] == 2

    def test_get_edges_and_counts_missing_conversation_key(self):
        """Test with data missing conversation key."""
        data = [{"other_key": "value"}]
        result = get_edges_and_counts(data)
        assert result == {}

    def test_get_edges_and_counts_empty_conversation(self):
        """Test with empty conversation."""
        data = [{"conversation": []}]
        result = get_edges_and_counts(data)
        assert result == {}


class TestBuildIntentGraph:
    """Test cases for build_intent_graph function."""

    def test_build_intent_graph_empty_data(self):
        """Test with empty data list."""
        data = []
        graph = build_intent_graph(data)
        assert isinstance(graph, nx.DiGraph)
        assert graph.number_of_nodes() == 0
        assert graph.number_of_edges() == 0

    def test_build_intent_graph_single_conversation(self):
        """Test with single conversation."""
        data = [
            {
                "conversation": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"},
                ]
            }
        ]
        graph = build_intent_graph(data)
        assert isinstance(graph, nx.DiGraph)
        assert graph.number_of_nodes() == 2
        assert graph.number_of_edges() == 1
        assert "Hello" in graph.nodes()
        assert "Hi there" in graph.nodes()
        assert graph.has_edge("Hello", "Hi there")

    def test_build_intent_graph_multiple_conversations(self):
        """Test with multiple conversations."""
        data = [
            {
                "conversation": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                ]
            },
            {
                "conversation": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                ]
            },
        ]
        graph = build_intent_graph(data)
        assert isinstance(graph, nx.DiGraph)
        assert graph.number_of_nodes() == 2
        assert graph.number_of_edges() == 1
        assert graph["Hello"]["Hi"]["weight"] == 2

    def test_build_intent_graph_complex_conversation(self):
        """Test with complex conversation."""
        data = [
            {
                "conversation": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"},
                    {"role": "user", "content": "How are you?"},
                    {"role": "assistant", "content": "I'm good, thanks!"},
                ]
            }
        ]
        graph = build_intent_graph(data)
        assert isinstance(graph, nx.DiGraph)
        assert graph.number_of_nodes() == 4
        assert graph.number_of_edges() == 3
        assert graph.has_edge("Hello", "Hi there")
        assert graph.has_edge("Hi there", "How are you?")
        assert graph.has_edge("How are you?", "I'm good, thanks!")

    def test_build_intent_graph_missing_conversation_key(self):
        """Test with data missing conversation key."""
        data = [{"other_key": "value"}]
        graph = build_intent_graph(data)
        assert isinstance(graph, nx.DiGraph)
        assert graph.number_of_nodes() == 0
        assert graph.number_of_edges() == 0
