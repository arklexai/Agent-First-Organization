"""Prompt templates for the orchestrator in the Arklex framework.

This module contains the prompt templates used by the orchestrator for various tasks,
including action selection, trajectory summarization, and planning. It includes
prompts for both zero-shot and few-shot scenarios, with templates for reasoning
about available tools and actions, and generating appropriate responses.

Key Components:
1. Action Constants
   - RESPOND_ACTION_NAME: Name of the response action
   - RESPOND_ACTION_FIELD_NAME: Field name for response content

2. Prompt Templates
   - REACT_INSTRUCTION: Base instruction for ReAct-based reasoning
   - PLANNER_REACT_INSTRUCTION_ZERO_SHOT: Zero-shot planning instruction
   - PLANNER_SUMMARIZE_TRAJECTORY_PROMPT: Trajectory summarization template
   - PLANNER_REACT_INSTRUCTION_FEW_SHOT: Few-shot planning instruction

Features:
- Structured prompt templates
- Support for both zero-shot and few-shot learning
- Tool and action reasoning
- Trajectory summarization
- Response formatting
- Task context integration

Usage:
    from arklex.orchestrator.prompts import REACT_INSTRUCTION

    # Format the instruction with context
    formatted_prompt = REACT_INSTRUCTION.format(
        conversation_record=history,
        available_tools=tools,
        task=current_task
    )
"""

# Action name constants
RESPOND_ACTION_NAME = "respond"  # Name of the response action
RESPOND_ACTION_FIELD_NAME = "content"  # Field name for response content

# Base ReAct instruction template
REACT_INSTRUCTION = """
# Instruction
You need to act as an agent that use a set of tools to help the user according to the policy.

# Conversation record
{conversation_record}

# Available tools
{available_tools}

Your generation should have exactly the following format:
Thought:
<A single line of reasoning to process the context and inform the decision making. Do not include extra lines.>
Action:
{{"name": <The name of the action>, "arguments": <The arguments to the action in json format>}}

You current task is:
{task}

Make the decision based on the current task, conversation record, and available tools. If the task has not been finished and available tools are helpful for the task, you should use the appropriate tool to finish it instead of directly give a response.

Thought:
"""

# Zero-shot planning instruction template
PLANNER_REACT_INSTRUCTION_ZERO_SHOT = """
# Instruction
Please act as an agent that selects the next appropriate action in a sequence of actions in order to satisfy the user's request.

# User message
{user_message}

# Available actions
{available_actions}

Your generation should have exactly the following format:
Thought:
<A single line of reasoning to process the context and inform the decision making. Do not include extra lines.>

Action:
{{"name": <The name of the action>, "arguments": <Any arguments that the action requires in json format>}}

If you must return a message to the user or ask them for more information, your response must do so using the action "{respond_action_name}", for example:
Action:
{{"name": "{respond_action_name}", "arguments": {{"content": <message to return to the user>}}}}

Never provide a response that is not in this format.

Your current task is:
{task}

Select the next action based on the current task, conversation record, and available actions.

Thought:
"""

# Trajectory summarization template
PLANNER_SUMMARIZE_TRAJECTORY_PROMPT = """
# Instruction
Please summarize the planning steps required to satisfy the user's request.
Your response must be formatted as a bulleted list where each line begins with a hyphen ("-"). Do not include any extraneous text.

# User message
{user_message}

# Available actions
To help determine what steps are required to satisfy the request, refer to the following descriptions of available actions: {resource_descriptions}

Your current task is:
{task}

Answer:
"""

# Few-shot planning instruction template with example
PLANNER_REACT_INSTRUCTION_FEW_SHOT = """
# Instruction
Please act as an agent that selects the next appropriate action in a sequence of actions in order to satisfy the user's request.

Your generation should have exactly the following format:
Thought:
<A single line of reasoning to process the context and inform the decision making. Do not include extra lines.>

Action:
{{"name": <The name of the action>, "arguments": <Any arguments that the action requires in json format>}}

If you must return a message to the user or ask them for more information, your response must do so using the action "{respond_action_name}", for example:
Action:
{{"name": "{respond_action_name}", "arguments": {{"content": <message to return to the user>}}}}

Never provide a response that is not in this format.

For example: 
---

Select the next action based on the current task, conversation record, and available actions.

# User message
Can you please retrieve my user details? My email is sample-email@arklex.ai.

# Available actions
{{'shopify-find_user_id_by_email-find_user_id_by_email': {{'name': 'shopify-find_user_id_by_email-find_user_id_by_email', 'type': 'tool', 'description': 'Find user id by email. If the user is not found, the function will return an error message.', 'parameters': [{{'user_email': {{'type': 'string', 'description': "The email of the user, such as 'something@example.com'."}}}}], 'required': ['user_email'], 'returns': {{'user_id': "The user id of the user. such as 'gid://shopify/Customer/13573257450893'."}}}}, 'shopify-get_user_details_admin-get_user_details_admin': {{'name': 'shopify-get_user_details_admin-get_user_details_admin', 'type': 'tool', 'description': 'Get the details of a user with Admin API.', 'parameters': [{{'user_id': {{'type': 'string', 'description': "The user id, such as 'gid://shopify/Customer/13573257450893'."}}}}, {{'limit': {{'type': 'string', 'description': 'Maximum number of entries to show.'}}}}], 'required': ['user_id'], 'returns': {{'user_details': 'The user details of the user. such as \'{{"firstName": "John", "lastName": "Doe", "email": "example@gmail.com"}}\'.'}}}}, 'FaissRAGWorker': {{'name': 'FaissRAGWorker', 'type': 'worker', 'description': "Answer the user's questions based on the company's internal documentations (unstructured text data), such as the policies, FAQs, and product information", 'parameters': [], 'required': [], 'returns': {{}}}}, 'shopify-get_cart-get_cart': {{'name': 'shopify-get_cart-get_cart', 'type': 'tool', 'description': 'Get cart information', 'parameters': [{{'cart_id': {{'type': 'str', 'description': "The cart id, such as 'gid://shopify/Cart/2938501948327'."}}}}, {{'limit': {{'type': 'string', 'description': 'Maximum number of entries to show.'}}}}], 'required': ['cart_id'], 'returns': {{'get_cart': 'The cart details of the cart.'}}}}, 'respond': {{'name': 'respond', 'type': 'worker', 'description': "Respond to the user if the user's request has been satisfied or if there is not enough information to do so.", 'parameters': [{{'content': {{'type': 'string', 'description': 'The message to return to the user.'}}}}], 'required': ['content'], 'returns': {{}}}}}}

Your current task is:


Thought:
To retrieve the user's details, I first need to find the user ID using the provided email address.

Action:
{{"name": "shopify-find_user_id_by_email-find_user_id_by_email", "arguments": {{"user_email": "sample-email@arklex.ai"}}}}

---

Select the next action based on the current task, conversation record, and available tools.

# User message
{user_message}

# Available actions
{available_actions}

Your current task is:
{task}

Thought:
"""
