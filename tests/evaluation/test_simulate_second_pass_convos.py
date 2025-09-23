"""Tests for simulate_second_pass_convos module."""

import pytest
import networkx as nx
from unittest.mock import patch, MagicMock
from arklex.evaluation.simulate_second_pass_convos import (
    sampling_paths,
    get_paths,
)


class TestSamplingPaths:
    """Test cases for sampling_paths function."""

    def test_sampling_paths_basic(self):
        """Test basic path sampling."""
        graph = nx.DiGraph()
        graph.add_edge("start", "middle")
        graph.add_edge("middle", "end")
        
        intents = ["intent1", "intent2"]
        result = sampling_paths("start", graph, 0, 3, intents)
        
        assert isinstance(result, list)
        assert len(result) > 0

    def test_sampling_paths_max_turns_reached(self):
        """Test when max_turns is reached."""
        graph = nx.DiGraph()
        graph.add_edge("start", "middle")
        graph.add_edge("middle", "end")
        
        intents = ["intent1", "intent2"]
        result = sampling_paths("start", graph, 0, 1, intents)
        
        assert isinstance(result, list)
        # Should return early due to max_turns limit

    def test_sampling_paths_no_neighbors(self):
        """Test with node having no neighbors."""
        graph = nx.DiGraph()
        graph.add_node("start")
        
        intents = ["intent1", "intent2"]
        result = sampling_paths("start", graph, 0, 3, intents)
        
        assert isinstance(result, list)
        assert len(result) == 0

    def test_sampling_paths_empty_graph(self):
        """Test with empty graph."""
        graph = nx.DiGraph()
        
        intents = ["intent1", "intent2"]
        result = sampling_paths("start", graph, 0, 3, intents)
        
        assert isinstance(result, list)
        assert len(result) == 0

    def test_sampling_paths_empty_intents(self):
        """Test with empty intents list."""
        graph = nx.DiGraph()
        graph.add_edge("start", "middle")
        
        intents = []
        result = sampling_paths("start", graph, 0, 3, intents)
        
        assert isinstance(result, list)
        assert len(result) == 0

    def test_sampling_paths_negative_path_length(self):
        """Test with negative path length."""
        graph = nx.DiGraph()
        graph.add_edge("start", "middle")
        
        intents = ["intent1", "intent2"]
        result = sampling_paths("start", graph, -1, 3, intents)
        
        assert isinstance(result, list)

    def test_sampling_paths_zero_max_turns(self):
        """Test with zero max_turns."""
        graph = nx.DiGraph()
        graph.add_edge("start", "middle")
        
        intents = ["intent1", "intent2"]
        result = sampling_paths("start", graph, 0, 0, intents)
        
        assert isinstance(result, list)
        assert len(result) == 0


class TestGetPaths:
    """Test cases for get_paths function."""

    def test_get_paths_basic(self):
        """Test basic path generation."""
        graph = nx.DiGraph()
        graph.add_edge("start", "middle")
        graph.add_edge("middle", "end")
        
        result = get_paths(graph, 2, 3)
        
        assert isinstance(result, list)
        assert len(result) <= 2  # Should not exceed num_paths

    def test_get_paths_empty_graph(self):
        """Test with empty graph."""
        graph = nx.DiGraph()
        
        result = get_paths(graph, 2, 3)
        
        assert isinstance(result, list)
        assert len(result) == 0

    def test_get_paths_zero_num_paths(self):
        """Test with zero num_paths."""
        graph = nx.DiGraph()
        graph.add_edge("start", "middle")
        
        result = get_paths(graph, 0, 3)
        
        assert isinstance(result, list)
        assert len(result) == 0

    def test_get_paths_zero_max_turns(self):
        """Test with zero max_turns."""
        graph = nx.DiGraph()
        graph.add_edge("start", "middle")
        
        result = get_paths(graph, 2, 0)
        
        assert isinstance(result, list)
        assert len(result) == 0

    def test_get_paths_single_node_graph(self):
        """Test with single node graph."""
        graph = nx.DiGraph()
        graph.add_node("start")
        
        result = get_paths(graph, 2, 3)
        
        assert isinstance(result, list)
        assert len(result) == 0

    def test_get_paths_complex_graph(self):
        """Test with complex graph."""
        graph = nx.DiGraph()
        graph.add_edge("start", "middle1")
        graph.add_edge("start", "middle2")
        graph.add_edge("middle1", "end")
        graph.add_edge("middle2", "end")
        
        result = get_paths(graph, 5, 3)
        
        assert isinstance(result, list)
        assert len(result) <= 5

    def test_get_paths_negative_num_paths(self):
        """Test with negative num_paths."""
        graph = nx.DiGraph()
        graph.add_edge("start", "middle")
        
        result = get_paths(graph, -1, 3)
        
        assert isinstance(result, list)
        assert len(result) == 0

    def test_get_paths_negative_max_turns(self):
        """Test with negative max_turns."""
        graph = nx.DiGraph()
        graph.add_edge("start", "middle")
        
        result = get_paths(graph, 2, -1)
        
        assert isinstance(result, list)
        assert len(result) == 0
