RESPOND_ACTION_NAME = "respond"
RESPOND_ACTION_FIELD_NAME = "content"

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

### REACT PLANNER PROMPTS

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

PLANNER_REACT_INSTRUCTION_ZERO_SHOT = """
# Instruction
Please act as an agent that selects the next appropriate action as a tool call in order to satisfy the user's request.

# User message
{user_message}

Your generation should have exactly the following format:
Thought:
<A single line of reasoning to process the context and inform the decision making. Do not include extra lines.>

Please call any tools that may be required to satisfy the user's request or accomplish the user's goal.

If you need to return a message to the user or ask them for more information, you must call the tool with the name '{respond_action_name}'.

Never provide a response that does not adhere to these guidelines.

Your current task is:
{task}

Select the next action as a tool call based on the current task and conversation record.

Thought:
"""

PLANNER_REACT_INSTRUCTION_FEW_SHOT = """
# Instruction
Please act as an agent that selects the next appropriate action in a sequence of actions in order to satisfy the user's request.

Your generation should have exactly the following format:
Thought:
<A single line of reasoning to process the context and inform the decision making. Do not include extra lines.>

Please call any tools that may be required to satisfy the user's request or accomplish the user's goal.

If you need to return a message to the user or ask them for more information, you must call the tool with the name '{respond_action_name}'.

Never provide a response that does not adhere to these guidelines.

For example: 
---

Select the next action based on the current task and conversation record.

# User message
Can you please retrieve my user details? My email is sample-email@arklex.ai.

Your current task is:
None

Thought:
To retrieve the user's details, I first need to find the user ID using the provided email address.

---

The above response should call the tool '"shopify-find_user_id_by_email-find_user_id_by_email' with the argument: {{"user_email": "sample-email@arklex.ai"}}.


Select the next action (as a tool call) based on the user message, current task, and conversation record.

# User message
{user_message}

Your current task is:
{task}

Thought:
"""

PLANNER_REASONING_INSTRUCTION_ZERO_SHOT = """
# Instruction
Please act as an agent that reasons about the next action required to satisfy the user's request. Consider the conversation trajectory and the information provided about the available tools/actions.

Your generation should have exactly the following format:
Thought:
<A single line of reasoning to process the context and inform the decision making. Do not include extra lines.>

Never provide a response that does not adhere to these guidelines.

# User message
{user_message}

# Available Actions
{available_actions}

Your current task is:
{task}

Thought:
"""

PLANNER_REASONING_INSTRUCTION_FEW_SHOT = """
# Instruction
Please act as an agent that reasons about the next action required to satisfy the user's request. Consider the conversation trajectory and the information provided about the available tools/actions.

Your generation should have exactly the following format:
Thought:
<A single line of reasoning to process the context and inform the decision making. Do not include extra lines.>

Never provide a response that does not adhere to these guidelines.

For example: 
---
# User message
Can you please retrieve my user details? My email is sample-email@arklex.ai.

# Available actions
[{{'name': 'shopify-find_user_id_by_email-find_user_id_by_email', 'type': 'tool', 'description': 'Find user id by email. If the user is not found, the function will return an error message.', 'parameters': [{{'user_email': {{'type': 'str', 'description': "The email of the user, such as 'something@example.com'."}}}}], 'required': ['user_email'], 'returns': {{'user_id': "The user id of the user. such as 'gid://shopify/Customer/13573257450893'."}}}}, {{'name': 'shopify-get_user_details_admin-get_user_details_admin', 'type': 'tool', 'description': 'Get the details of a user with Admin API.', 'parameters': [{{'user_id': {{'type': 'str', 'description': "The user id, such as 'gid://shopify/Customer/13573257450893'."}}}}], 'required': ['user_id'], 'returns': {{'user_details': 'The user details of the user. such as \'{{"firstName": "John", "lastName": "Doe", "email": "example@gmail.com"}}\'.', 'pageInfo': 'Current pageInfo object, such as  "{{\'endCursor\': \'eyJsYXN0X2lkIjo3Mjk2NTgxODk0MjU3LCJsYXN0X3ZhbHVlIjoiNzI5NjU4MTg5NDI1NyJ9\', \'hasNextPage\': True, \'hasPreviousPage\': False, \'startCursor\': \'eyJsYXN0X2lkIjo3Mjk2NTgwODQ1NjgxLCJsYXN0X3ZhbHVlIjoiNzI5NjU4MDg0NTY4MSJ9\'}}"'}}}}, {{'name': 'shopify-get_order_details-get_order_details', 'type': 'tool', 'description': 'Get the status and details of an order.', 'parameters': ...}}}}]
Your current task is:
None

Thought:
To retrieve the user's details, I first need to find the user ID using the provided email address.
---

# User message
{user_message}

# Available Actions
{available_actions}

Your current task is:
{task}

Thought:
"""

PLANNER_TOOL_CALL_INSTRUCTION_ZERO_SHOT = """
# Instruction
Please act as an agent that selects the next appropriate action as a tool call in order to satisfy the user's request.

Please call any tools that may be required to satisfy the user's request or accomplish the user's goal.

If you need to return a message to the user or ask them for more information, you must call the tool with the name '{respond_action_name}'.

# User message
{user_message}

Your current task is:
{task}

# Thought
{thought}

Select the next action as a tool call based on the current task, conversation record, and the most recent thought.
"""