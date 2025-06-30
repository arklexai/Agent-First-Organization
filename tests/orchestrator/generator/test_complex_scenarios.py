"""Tests for complex scenarios in task generation and optimization.

This module provides comprehensive tests for advanced task generation,
optimization strategies, nested workflows, and sophisticated resource allocation.
"""

import json

import pytest

from arklex.orchestrator.generator.core.generator import Generator
from arklex.orchestrator.generator.formatting.task_graph_formatter import (
    TaskGraphFormatter,
)
from arklex.orchestrator.generator.tasks.best_practice_manager import (
    BestPracticeManager,
)
from arklex.orchestrator.generator.tasks.task_generator import TaskGenerator
from tests.orchestrator.generator.test_mock_models import (
    MockLanguageModelWithCustomResponses,
    create_mock_model_for_best_practices,
    create_mock_model_for_task_generation,
)

# --- Fixtures ---


@pytest.fixture
def advanced_config() -> dict:
    """Create advanced configuration for complex scenarios."""
    return {
        "role": "Advanced Customer Service Assistant",
        "user_objective": "Provide comprehensive customer support with AI-powered insights",
        "builder_objective": "Create an intelligent customer service system with predictive analytics",
        "domain": "E-commerce and Customer Support",
        "intro": "Advanced e-commerce platform with AI-powered customer service capabilities",
        "task_docs": [],
        "rag_docs": [
            {
                "name": "Customer Behavior Database",
                "content": "Historical customer interaction data and patterns",
            },
            {
                "name": "Product Recommendation Engine",
                "content": "AI-powered product recommendation algorithms",
            },
        ],
        "workers": [
            {"name": "MessageWorker", "id": "msg_worker_1"},
            {"name": "FaissRAGWorker", "id": "rag_worker_1"},
            {"name": "SearchWorker", "id": "search_worker_1"},
            {"name": "AnalyticsWorker", "id": "analytics_worker_1"},
            {"name": "PredictionWorker", "id": "prediction_worker_1"},
        ],
        "tools": [
            {
                "name": "CustomerAnalytics",
                "description": "Advanced customer analytics",
                "path": "mock_path",
                "id": "customer_analytics_1",
            },
            {
                "name": "PredictiveModeling",
                "description": "Predictive modeling tools",
                "path": "mock_path",
                "id": "predictive_modeling_1",
            },
            {
                "name": "BehavioralAnalysis",
                "description": "Customer behavior analysis",
                "path": "mock_path",
                "id": "behavioral_analysis_1",
            },
            {
                "name": "RecommendationEngine",
                "description": "AI recommendation engine",
                "path": "mock_path",
                "id": "recommendation_engine_1",
            },
            {
                "name": "SentimentAnalysis",
                "description": "Sentiment analysis tools",
                "path": "mock_path",
                "id": "sentiment_analysis_1",
            },
        ],
    }


@pytest.fixture
def always_valid_mock_model() -> MockLanguageModelWithCustomResponses:
    """Create a mock model that always returns valid, non-empty task list."""
    model = create_mock_model_for_task_generation()
    valid_task = '[{"id": "task_1", "name": "Test Task", "description": "A test task", "steps": [{"description": "Step 1"}]}]'
    model.generate = lambda messages: type("Mock", (), {"content": valid_task})()
    model.invoke = lambda messages: type("Mock", (), {"content": valid_task})()
    return model


@pytest.fixture
def patched_advanced_config(advanced_config: dict) -> dict:
    """Create advanced config with tools patched to avoid import errors."""
    config = advanced_config.copy()
    config["tools"] = []
    return config


@pytest.fixture
def ai_powered_mock_model() -> MockLanguageModelWithCustomResponses:
    """Create a mock model configured for AI-powered task generation scenarios."""
    ai_model = MockLanguageModelWithCustomResponses()
    ai_model.add_response(
        "AI-powered",
        """[
            {
                "intent": "User needs AI-powered customer insights",
                "task": "AI Customer Analytics"
            },
            {
                "intent": "User wants predictive recommendations",
                "task": "Predictive Recommendation Engine"
            },
            {
                "intent": "User seeks behavioral analysis",
                "task": "Customer Behavior Analysis"
            }
        ]""",
    )
    ai_model.call_count = 1
    ai_model.generate = lambda messages: type(
        "Mock",
        (),
        {
            "content": '[{"id": "ai_task_1", "name": "AI Customer Analytics", "description": "Analyze customer behavior using AI algorithms", "steps": [{"description": "Step 1"}]}]'
        },
    )()
    return ai_model


@pytest.fixture
def optimization_config() -> dict:
    """Create configuration for optimization testing."""
    return {
        "role": "Optimization System",
        "user_objective": "Optimize task execution and resource allocation",
        "workers": [{"name": "DataWorker"}, {"name": "AnalyticsWorker"}],
        "tools": [
            {"name": "DataTool", "path": "mock_path"},
            {"name": "AnalyticsTool", "path": "mock_path"},
        ],
    }


@pytest.fixture
def optimization_mock_model() -> MockLanguageModelWithCustomResponses:
    """Create a mock model for optimization testing."""
    return create_mock_model_for_best_practices()


@pytest.fixture
def ai_pipeline_mock_model() -> MockLanguageModelWithCustomResponses:
    """Create a mock model for AI pipeline integration testing."""
    ai_model = create_mock_model_for_task_generation()
    ai_model.add_response(
        "task",
        """[
            {
                "task": "AI Customer Analytics",
                "intent": "AI-powered customer analysis",
                "description": "Analyze customer behavior using AI algorithms"
            },
            {
                "task": "Predictive Engine",
                "intent": "Predictive recommendations",
                "description": "Generate predictive recommendations for customers"
            }
        ]""",
    )
    ai_model.add_response(
        "default",
        """[
            {
                "task": "AI Customer Analytics",
                "intent": "AI-powered customer analysis",
                "description": "Analyze customer behavior using AI algorithms"
            },
            {
                "task": "Predictive Engine",
                "intent": "Predictive recommendations",
                "description": "Generate predictive recommendations for customers"
            }
        ]""",
    )
    return ai_model


# --- Test Classes ---


class TestAdvancedTaskGeneration:
    """Test advanced task generation scenarios."""

    def test_multi_layered_task_generation(
        self,
        patched_advanced_config: dict,
        always_valid_mock_model: MockLanguageModelWithCustomResponses,
    ) -> None:
        """Test generation of multi-layered task hierarchies."""
        generator = TaskGenerator(
            model=always_valid_mock_model,
            role=patched_advanced_config["role"],
            user_objective=patched_advanced_config["user_objective"],
            instructions="Generate complex multi-layered tasks",
            documents=json.dumps(patched_advanced_config["task_docs"]),
        )
        tasks = generator.generate_tasks(patched_advanced_config["intro"])
        assert len(tasks) > 0
        task_ids = [task.get("id") for task in tasks]
        for task in tasks:
            if task.get("dependencies"):
                for dep in task["dependencies"]:
                    assert dep in task_ids

    def test_ai_powered_task_generation(
        self,
        patched_advanced_config: dict,
        ai_powered_mock_model: MockLanguageModelWithCustomResponses,
    ) -> None:
        """Test AI-powered task generation with advanced features."""
        generator = TaskGenerator(
            model=ai_powered_mock_model,
            role=patched_advanced_config["role"],
            user_objective=patched_advanced_config["user_objective"],
            instructions="Generate AI-powered tasks",
            documents=json.dumps(patched_advanced_config["task_docs"]),
        )
        tasks = generator.generate_tasks(patched_advanced_config["intro"])
        assert len(tasks) > 0
        task_names = [task.get("name", "").lower() for task in tasks]
        ai_keywords = ["ai", "predictive", "analytics", "behavior", "recommendation"]
        assert any(
            any(keyword in name for keyword in ai_keywords) for name in task_names
        )

    def test_complex_dependency_chains(self, advanced_config: dict) -> None:
        """Test complex dependency chain resolution."""
        formatter = TaskGraphFormatter(
            role=advanced_config["role"],
            user_objective=advanced_config["user_objective"],
            allow_nested_graph=True,
        )
        complex_tasks = [
            {
                "id": "data_collection",
                "name": "Data Collection",
                "description": "Collect data from multiple sources",
                "dependencies": [],
                "steps": [{"description": "Collect data"}],
            },
            {
                "id": "data_processing",
                "name": "Data Processing",
                "description": "Process and clean collected data",
                "dependencies": ["data_collection"],
                "steps": [{"description": "Process data"}],
            },
            {
                "id": "analysis",
                "name": "Analysis",
                "description": "Perform statistical analysis",
                "dependencies": ["data_processing"],
                "steps": [{"description": "Analyze data"}],
            },
            {
                "id": "generate_insights",
                "name": "Generate Insights",
                "description": "Generate insights and recommendations",
                "dependencies": ["analysis"],
                "steps": [{"description": "Generate insights"}],
            },
        ]
        result = formatter.format_task_graph(complex_tasks)
        assert "nodes" in result
        assert "edges" in result
        assert len(result["nodes"]) >= 4
        assert len(result["edges"]) >= 3

    def test_adaptive_task_generation(
        self,
        patched_advanced_config: dict,
        always_valid_mock_model: MockLanguageModelWithCustomResponses,
    ) -> None:
        """Test adaptive task generation with changing objectives."""
        generator = TaskGenerator(
            model=always_valid_mock_model,
            role=patched_advanced_config["role"],
            user_objective=patched_advanced_config["user_objective"],
            instructions="Generate adaptive tasks",
            documents=json.dumps(patched_advanced_config["task_docs"]),
        )
        objectives = [
            "Handle product browsing",
            "Process purchases",
        ]
        all_task_names = set()
        for obj in objectives:
            tasks = generator.generate_tasks(obj)
            all_task_names.update(task.get("name") for task in tasks)
        assert len(all_task_names) >= 1


class TestAdvancedOptimization:
    """Test advanced optimization strategies."""

    def test_resource_optimization(
        self,
        optimization_config: dict,
        optimization_mock_model: MockLanguageModelWithCustomResponses,
    ) -> None:
        """Test resource optimization in practice generation."""
        practice_manager = BestPracticeManager(
            model=optimization_mock_model,
            role="Data Processing System",
            user_objective="Optimize data processing workflows",
            workers=optimization_config["workers"],
            tools=optimization_config["tools"],
        )
        task_with_resources = [
            {
                "id": "data_processing",
                "name": "Data Processing",
                "description": "Process and analyze data efficiently",
                "steps": [
                    {"description": "Load data", "resource": "DataWorker"},
                    {"description": "Process data", "resource": "AnalyticsWorker"},
                ],
            }
        ]
        practices = practice_manager.generate_best_practices(task_with_resources)
        assert isinstance(practices, list)

    def test_performance_optimization(
        self,
        optimization_config: dict,
        optimization_mock_model: MockLanguageModelWithCustomResponses,
    ) -> None:
        """Test performance optimization strategies."""
        practice_manager = BestPracticeManager(
            model=optimization_mock_model,
            role=optimization_config["role"],
            user_objective=optimization_config["user_objective"],
            workers=optimization_config["workers"],
            tools=optimization_config["tools"],
        )
        performance_tasks = [
            {
                "id": "perf_task",
                "name": "Performance Task",
                "description": "Optimize task performance",
                "steps": [{"description": "Optimize step"}],
            }
        ]
        practices = practice_manager.generate_best_practices(performance_tasks)
        assert isinstance(practices, list)

    def test_multi_objective_optimization(
        self,
        optimization_config: dict,
        optimization_mock_model: MockLanguageModelWithCustomResponses,
    ) -> None:
        """Test multi-objective optimization scenarios."""
        practice_manager = BestPracticeManager(
            model=optimization_mock_model,
            role=optimization_config["role"],
            user_objective=optimization_config["user_objective"],
            workers=optimization_config["workers"],
            tools=optimization_config["tools"],
        )
        multi_objective_tasks = [
            {
                "id": "multi_task",
                "name": "Multi-objective Task",
                "description": "Balance multiple objectives",
                "steps": [{"description": "Balance objectives"}],
            }
        ]
        practices = practice_manager.generate_best_practices(multi_objective_tasks)
        assert isinstance(practices, list)


class TestNestedWorkflowScenarios:
    """Test nested workflow scenarios."""

    def test_deep_nested_workflows(self, advanced_config: dict) -> None:
        """Test deeply nested workflow structures."""
        formatter = TaskGraphFormatter(
            role=advanced_config["role"],
            user_objective=advanced_config["user_objective"],
            allow_nested_graph=True,
        )
        nested_tasks = [
            {
                "id": "main_workflow",
                "name": "Main Workflow",
                "description": "Main workflow with nested sub-workflows",
                "steps": [
                    {"description": "Step 1", "step_id": "step_1"},
                    {"description": "Step 2", "step_id": "step_2"},
                ],
                "resource": {"name": "NestedGraph"},
            }
        ]
        result = formatter.format_task_graph(nested_tasks)
        assert "nodes" in result
        assert "edges" in result
        assert len(result["nodes"]) > 0

    def test_parallel_nested_workflows(self, advanced_config: dict) -> None:
        """Test parallel nested workflow execution."""
        formatter = TaskGraphFormatter(
            role=advanced_config["role"],
            user_objective=advanced_config["user_objective"],
            allow_nested_graph=True,
        )
        parallel_tasks = [
            {
                "id": "parallel_workflow_1",
                "name": "Parallel Workflow 1",
                "description": "First parallel workflow",
                "steps": [{"description": "Step 1"}],
                "resource": {"name": "NestedGraph"},
            },
            {
                "id": "parallel_workflow_2",
                "name": "Parallel Workflow 2",
                "description": "Second parallel workflow",
                "steps": [{"description": "Step 2"}],
                "resource": {"name": "NestedGraph"},
            },
        ]
        result = formatter.format_task_graph(parallel_tasks)
        assert "nodes" in result
        assert "edges" in result
        assert len(result["nodes"]) > 0


class TestAdvancedIntegrationScenarios:
    """Test advanced integration scenarios."""

    def test_full_ai_pipeline_integration(
        self,
        patched_advanced_config: dict,
        always_valid_mock_model: MockLanguageModelWithCustomResponses,
        ai_pipeline_mock_model: MockLanguageModelWithCustomResponses,
    ) -> None:
        """Test full AI pipeline integration with comprehensive mock model."""
        generator = Generator(
            config=patched_advanced_config,
            model=always_valid_mock_model,
            interactable_with_user=False,
        )
        task_graph = generator.generate()
        assert "nodes" in task_graph
        assert "edges" in task_graph
        assert "tasks" in task_graph
        assert len(task_graph["nodes"]) > 0
        assert isinstance(task_graph["edges"], list)
        assert len(task_graph["tasks"]) > 0

    def test_scalable_architecture_integration(
        self,
        patched_advanced_config: dict,
        always_valid_mock_model: MockLanguageModelWithCustomResponses,
    ) -> None:
        """Test scalable architecture integration."""
        scalable_config = patched_advanced_config.copy()
        scalable_config["workers"].extend(
            [
                {"name": "ScalableWorker1", "id": "scalable_worker_1"},
                {"name": "ScalableWorker2", "id": "scalable_worker_2"},
            ]
        )
        generator = Generator(
            config=scalable_config,
            model=always_valid_mock_model,
            interactable_with_user=False,
        )
        task_graph = generator.generate()
        assert "nodes" in task_graph
        assert "edges" in task_graph
        assert "tasks" in task_graph
        assert len(task_graph["nodes"]) > 0
        assert isinstance(task_graph["edges"], list)
        assert len(task_graph["tasks"]) > 0

    def test_real_time_processing_integration(
        self,
        patched_advanced_config: dict,
        always_valid_mock_model: MockLanguageModelWithCustomResponses,
    ) -> None:
        """Test real-time processing integration."""
        realtime_config = patched_advanced_config.copy()
        realtime_config["workers"].extend(
            [
                {"name": "RealTimeWorker", "id": "realtime_worker_1"},
            ]
        )
        generator = Generator(
            config=realtime_config,
            model=always_valid_mock_model,
            interactable_with_user=False,
        )
        task_graph = generator.generate()
        assert "nodes" in task_graph
        assert "edges" in task_graph
        assert "tasks" in task_graph
        assert len(task_graph["nodes"]) > 0
        assert isinstance(task_graph["edges"], list)
        assert len(task_graph["tasks"]) > 0
