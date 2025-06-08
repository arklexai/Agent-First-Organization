"""Evaluation package for the Arklex framework.

This package provides tools and utilities for evaluating the performance and effectiveness
of the Arklex framework. It includes functionality for conversation simulation, user profile
generation, document processing, and metrics extraction. The package supports both first-pass
and second-pass conversation evaluation, with capabilities for synthetic data generation,
goal completion tracking, and performance analysis.

Key Components:
- Conversation Simulation:
  - First-pass conversation simulation (simulate_first_pass_convos.py)
  - Second-pass conversation simulation (simulate_second_pass_convos.py)
  - Conversation information extraction (extract_conversation_info.py)
- User Profile Management:
  - User profile generation (build_user_profiles.py)
  - User attributes configuration (user_attributes.json)
- Document Processing:
  - Document retrieval and processing (get_documents.py)
  - ChatGPT utilities for evaluation (chatgpt_utils.py)
- Data Management:
  - Data directory for storing evaluation data and results
  - README with evaluation guidelines and procedures

Key Features:
- Comprehensive conversation simulation
- Synthetic user profile generation
- Document processing and analysis
- Performance metrics extraction
- Goal completion tracking
- Automated evaluation workflows

Usage:
    # Import evaluation utilities as needed
    from arklex.evaluation import (
        simulate_first_pass_convos,
        simulate_second_pass_convos,
        extract_conversation_info,
        build_user_profiles,
        get_documents,
        chatgpt_utils
    )

    # Run evaluation workflows
    # 1. Generate user profiles
    # 2. Simulate conversations
    # 3. Extract and analyze metrics
    # 4. Generate evaluation reports
"""

# Note: This package contains tools for comprehensive evaluation of the Arklex framework,
# including conversation simulation, user profile generation, and performance analysis.
