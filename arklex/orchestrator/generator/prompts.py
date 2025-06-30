"""Prompt templates for task graph generation in the Arklex framework.

This module contains the prompt templates used by the task graph generator to create
and manage task hierarchies. It includes prompts for generating tasks, reusable subtasks,
and best practices for task organization.

Key Components:
- Task Generation Prompts: Templates for creating main tasks and subtasks
- Reusable Task Prompts: Templates for identifying and defining shared subtasks
- Best Practice Prompts: Templates for task decomposition and optimization
- Task Validation Prompts: Templates for verifying task structure and completeness

Features:
- Structured JSON output formatting
- Hierarchical task organization
- Reusable task identification
- Task decomposition guidance
- Resource utilization optimization
- Example-based learning
- Clear reasoning processes
- Comprehensive documentation

Usage:
    from arklex.orchestrator.generator.prompts import (
        generate_tasks_sys_prompt,
        generate_reusable_tasks_sys_prompt,
        check_best_practice_sys_prompt
    )

    # Format task generation prompt
    task_prompt = generate_tasks_sys_prompt.format(
        role="customer_service",
        u_objective="Handle customer inquiries",
        intro="E-commerce platform information",
        docs="Product documentation",
        instructions="Task generation guidelines",
        existing_tasks="Current task list"
    )

    # Format reusable task prompt
    reusable_prompt = generate_reusable_tasks_sys_prompt.format(
        role="customer_service",
        u_objective="Handle customer inquiries",
        intro="Platform overview",
        tasks="Main task list",
        docs="Available documentation",
        instructions="Task guidelines",
        example_conversations="Sample interactions"
    )

    # Format best practice check prompt
    practice_prompt = check_best_practice_sys_prompt.format(
        task="Handle customer inquiry",
        level=1,
        resources="Available workers and tools"
    )
"""

# Intent Generation Prompt Template
generate_intents_sys_prompt = """
Your task is to generate a user-facing intent for a given task.
The intent should be a concise and clear description of what the user wants to achieve.
Base the intent on the task's name and description.

Here are some examples:
- Task Name: "Provide product specifications"
  - Intent: "User inquires about product specifications"
- Task Name: "Process a return"
  - Intent: "User wants to process a return"
- Task Name: "Schedule a demo"
  - Intent: "User wants to schedule a product demonstration"

Now, generate an intent for the following task:
Task Name: {task_name}
Task Description: {task_description}

Return the response in JSON format, like this:
```json
{{
    "intent": "Generated intent goes here"
}}
```
"""

# Task Generation Prompt Template
generate_tasks_sys_prompt = """The builder plans to create a chatbot designed to fulfill user's objectives. 
Given the role of the chatbot, along with any introductory information and detailed documentation (if available), your task is to identify the specific, distinct tasks that a chatbot should handle based on the user's intent. You are also given a list of existing tasks with user's intent. You must not return tasks that deal the same existing user's intent. All tasks should not overlap or depend on each other and must address different aspects of the user's goals. Ensure that each task represents a unique user intent and that they can operate separately. Moreover, you are given the instructions that you must follow.

Return the response in JSON format with only the high-level tasks. Do not break down tasks into steps at this stage.

For Example:

Builder's prompt: The builder want to create a chatbot - Customer Service Assistant. The customer service assistant typically handles tasks such as answering customer inquiries, making product recommendations, assisting with orders, processing returns and exchanges, supporting billing and payments, addressing complaints, and managing customer accounts.
Builder's Information: Amazon.com is a large e-commerce platform that sells a wide variety of products, ranging from electronics to groceries.
Builder's documentations: 
https://www.amazon.com/
Holiday Deals
Disability Customer Support
Same-Day Delivery
Medical Care
Customer Service
Amazon Basics
Groceries
Prime
Buy Again
New Releases
Pharmacy
Shop By Interest
Amazon Home
Amazon Business
Subscribe & Save
Livestreams
luwanamazon's Amazon.com
Best Sellers
Household, Health & Baby Care
Sell
Gift Cards

https://www.amazon.com/bestsellers
Any Department
Amazon Devices & Accessories
Amazon Renewed
Appliances
Apps & Games
Arts, Crafts & Sewing
Audible Books & Originals
Automotive
Baby
Beauty & Personal Care
Books
Camera & Photo Products
CDs & Vinyl
Cell Phones & Accessories
Clothing, Shoes & Jewelry
Collectible Coins
Computers & Accessories
Digital Educational Resources
Digital Music
Electronics
Entertainment Collectibles
Gift Cards
Grocery & Gourmet Food
Handmade Products
Health & Household
Home & Kitchen
Industrial & Scientific
Kindle Store
Kitchen & Dining
Movies & TV
Musical Instruments
Office Products
Patio, Lawn & Garden
Pet Supplies
Software
Sports & Outdoors
Sports Collectibles
Tools & Home Improvement
Toys & Games
Unique Finds
Video Games

Existing tasks:
Answer:
```json
[
    {{
        "intent": "User want to do product search and discovery",
        "task": "Provide help in Product Search and Discovery"
    }},
    {{
        "intent": "User ask for billing and payment support",
        "task": "Provide help in billing and payment support"
    }}
]
```

Builder's prompt: The builder want to create a chatbot - {role}. {u_objective}
Builder's information: {intro}
Builder's documentations: 
{docs}
Builder's instructions: 
{instructions}
Existing tasks:
{existing_tasks}
Reasoning Process:
"""

# Reusable Task Generation Prompt Template
generate_reusable_tasks_sys_prompt = """
The builder wants to create a chatbot with the following information:
Role of the Chatbot: {role}
User's Objective: {u_objective}
Builder's Introductory Information: {intro}
Builder's Tasks: {tasks}
Builder's Documentation (if any): {docs}
Instructions that you must follow: {instructions}
Here are some example conversations to help you understand how the bot's interactions should look. For each task, consider only the relevant parts of the example conversations: {example_conversations}
Your tasks is:
Identify Shared Subtasks: Based on the chatbot's role, any introductory information, available documentation, and the overall task set, identify and define subtasks that: Are essential to multiple tasks, Are granular, independent, and reusable, Can be logically grouped as recurring procedures.
Analyze Each Task: Break down each task into its smallest meaningful steps.
Exclude Simple or Overly Specific Procedures: Only include subtasks that represent more complex or significant actions and are relevant across multiple tasks. Exclude those that are overly simplistic or too task-specific.
Name Each Subtask Clearly: Use descriptive names for the subtasks.
Describe the Subtask's Purpose and Steps: Provide a clear and detailed definition for each subtask, along with a detailed outline of the steps it involves.
Maintain Independence: Subtasks should be self-contained, modular, and applicable in different contexts.
Return the Response in JSON Format
The JSON structure should include a clear name, description, and a steps (or next) hierarchy that outlines the logical flow.
The description should be detailed, so that one can understand clearly what the task does.
For each node in the hierarchy:
- Include a "task" field describing its function.
- Include a "next" array of child nodes. If a node is a leaf, set "next" to an empty array ([]).
Expected Answer Format (Example):
```json
[
    {{
        "name": "User Authentication",
        "description": "There are two types of user identity verification: via email or using name and ZIP code. This function first prompts the user to choose a verification method, then calls the corresponding authentication function accordingly.",
        "steps": {{
            "task": "Ask user for credentials; can authenticate via email or name + zip code.",
            "next": [
                {{
                    "task": "Locate user ID in the system using email.",
                    "next": []
                }},
                {{
                    "task": "Locate user ID in the system using name + zip code.",
                    "next": []
                }}
            ]
        }}
    }},
    {{
        "name": "Order Status Validation",
        "description": "This task is used to check the status of an order. It prompts the user for the order ID and retrieves the order details to determine the status. Finally, based on the allowed states for the requested action, it either proceeds or informs the user that they do not have permission.",
        "steps": [
            {{
                "task": "Ask user for order id.",
                "next": [
                    {{
                        "task": "Retrieve the order details.",
                        "next": [
                            {{
                                "task": "Verify if the order is in an allowed state for the requested action.",
                                "next": [
                                    {{
                                        "task": "If yes, continue with the requested action.",
                                        "next": []
                                    }},
                                    {{
                                        "task": "If not, inform the user and deny the request.",
                                        "next": []
                                    }}
                                ]
                            }}
                        ]
                    }}
                ]
            }}
        ]
    }}
]
```
Use this structure as a reference when defining and returning your own set of reusable subtasks in JSON.
"""

# Best Practice Check Prompt Template
check_best_practice_sys_prompt = """You are a userful assistance to detect if the current task needs to be further decomposed if it cannot be solved by the provided resources. Specifically, the task is positioned on a tree structure and is associated with a level. Based on the task and the current node level of the task on the tree, please output Yes if it needs to be decomposed; No otherwise meaning it is a singular task that can be handled by the resource and does not require task decomposition. Please also provide explanations for your choice. 

Here are some examples:
Task: The current task is Provide help in Product Search and Discovery. The current node level of the task is 1. 
Resources: 
MessageWorker: The worker responsible for interacting with the user with predefined responses,
RAGWorker: Answer the user's questions based on the company's internal documentations, such as the policies, FAQs, and product information,
ProductWorker: Access the company's database to retrieve information about products, such as availability, pricing, and specifications,
UserProfileWorker: Access the company's database to retrieve information about the user's preferences and history

Reasoning: This task is a high-level task that involves multiple sub-tasks such as asking for user's preference, providing product recommendations, answering questions about product or policy, and confirming user selections. Each sub-task requires different worker to complete. It will use MessageWorker to ask user's preference, then use ProductWorker to search for the product, finally make use of RAGWorker to answer user's question. So, it requires multiple interactions with the user and access to various resources. Therefore, it needs to be decomposed into smaller sub-tasks to be effectively handled by the assistant.
Answer: 
```json
{{
    "answer": "Yes"
}}
```

Task: The current task is booking a broadway show ticket. The current node level of the task is 1.
Resources:
DataBaseWorker: Access the company's database to retrieve information about ticket availability, pricing, and seating options. It will handle the booking process, which including confirming the booking details and providing relevant information. It can also handle the cancel process.
MessageWorker: The worker responsible for interacting with the user with predefined responses,
RAGWorker: Answer the user's questions based on the company's internal documentations, such as the policies, FAQs, and product information.

Reasoning: This task involves a single high-level action of booking a ticket for a broadway show. The task can be completed by accessing the database to check availability, pricing, and seating options, interacting with the user to confirm the booking details, and providing relevant information. Since it is a singular task that can be handled by the single resource without further decomposition, the answer is No.
Answer: 
```json
{{
    "answer": "No"
}}
```

Task: The current task is {task}. The current node level of the task is {level}.
Resources: {resources}
Reasoning:
"""


generate_best_practice_sys_prompt = """Given the background information about the chatbot, the task it needs to handle, and the available resources, your task is to generate a step-by-step best practice for addressing this task. Each step should represent a distinct interaction with the user, where the next step builds upon the user's response. Avoid breaking down sequences of internal worker actions within a single turn into multiple steps. Return the answer in JSON format. Only use resources listed in the input, resources listed in the Example may not be available. Moreover, you are given the instructions that you must follow

For example:
Background: The builder want to create a chatbot - Customer Service Assistant. The customer service assistant typically handles tasks such as answering customer inquiries, making product recommendations, assisting with orders, processing returns and exchanges, supporting billing and payments, addressing complaints, and managing customer accounts.

Task: Provide help in Product Search and Discovery

Resources:
MessageWorker: The worker responsible for interacting with the user with predefined responses,
RAGWorker: Answer the user's questions based on the company's internal documentations, such as the policies, FAQs, and product information,
ProductWorker: Access the company's database to retrieve information about products, such as availability, pricing, and specifications,
UserProfileWorker: Access the company's database to retrieve information about the user's preferences and history.

Instructions:
you should inquire specific preference before searching for the products that the user might want

Thought: To help users find products effectively, the assistant should first get context information about the customer from CRM, such as purchase history, demographic information, preference metadata, inquire about specific preferences or requirements (e.g., brand, features, price range) specific for the request. Second, based on the user's input, the assistant should provide personalized product recommendations. Third, the assistant should ask if there is anything not meet their goals. Finally, the assistant should confirm the user's selection, provide additional information if needed, and assist with adding the product to the cart or wish list.
Answer:
```json
[
    {{
      "step": 1,
      "task": "Retrieve the information about the customer and Inquire about specific preferences or requirements (e.g., brand, features, price range)."
    }},
    {{
      "step": 2,
      "task": "Search for the products that match user's preference and provide a curated list of products that match the user's criteria."
    }},
    {{
      "step": 3,
      "task": "Ask if the user would like to see more options or has any specific preferences."
    }},
    {{
      "step": 4,
      "task": "Confirm if the user is ready to proceed with a purchase or needs more help."
    }},
    {{
      "step": 5,
      "task": "Provide instructions for completing the purchase or next steps."
    }}
]
```

Background: The builder want to create a chatbot - {role}. {u_objective}
Task: {task}
Resources: {resources}
Instructions: {instructions}
Here are some example conversations to help you understand how the bot's interactions should look. For each task, consider only the relevant parts of the example conversations: {example_conversations}
Thought:
"""


# remove_duplicates_sys_prompt = """The builder plans to create a chatbot designed to fulfill user's objectives. Given the tasks and corresponding steps that the chatbot should handle, your task is to identify and remove any duplicate steps under each task that are already covered by other tasks. Ensure that each step is unique within the overall set of tasks and is not redundantly assigned. Return the response in JSON format.

# Tasks: {tasks}
# Answer:
# """

embed_builder_obj_sys_prompt = """The builder plans to create an assistant designed to provide services to users. Given the best practices for addressing a specific task and the builder's objectives, your task is to refine the steps to ensure they embed the objectives within each task. Return the answer in JSON format.

For example:
Best Practice: 
[
    {{
        "step": 1,
        "task": "Retrieve the information about the customer and Inquire about specific preferences or requirements (e.g., brand, features, price range)."
    }},
    {{
        "step": 2,
        "task": "Provide a curated list of products that match the user's criteria."
    }},
    {{
        "step": 3,
        "task": "Ask if the user would like to see more options or has any specific preferences."
    }},
    {{
        "step": 4,
        "task": "Confirm if the user is ready to proceed with a purchase or needs more help."
    }},
    {{
        "step": 5,
        "task": "Provide instructions for completing the purchase or next steps."
    }}
]
Build's objective: The customer service assistant helps in persuading customer to sign up the Prime membership.
Answer:
```json
[
    {{
        "step": 1,
        "task": "Retrieve the information about the customer from CRM and Inquire about specific preferences or requirements (e.g., brand, features, price range)."
    }},
    {{
        "step": 2,
        "task": "Provide a curated list of products that match the user's criteria."
    }},
    {{
        "step": 3,
        "task": "Ask if the user would like to see more options or has any specific preferences."
    }},
    {{
        "step": 4,
        "task": "Confirm if the user is ready to proceed with a purchase or needs more help."
    }},
    {{
        "step": 5,
        "task": "Persuade the user to sign up for the Prime membership."
    }}
]
```

Best Practice: {best_practice}
Build's objective: {b_objective}
Answer:
"""


embed_resources_sys_prompt = """The builder plans to create an assistant designed to provide services to users. Given the best practices for addressing a specific task, and the available resources, your task is to map the steps with the resources. The response should include only the most suitable resource used for each step and example responses, if applicable. Return the answer in JSON format. Do not add any comment on the answer.

For example:
Best Practice: 
[
    {{
        "step": 1,
        "task": "Retrieve the information about the customer from CRM and Inquire about specific preferences or requirements (e.g., brand, features, price range)."
    }},
    {{
        "step": 2,
        "task": "Provide a curated list of products that match the user's criteria."
    }},
    {{
        "step": 3,
        "task": "Ask if the user would like to see more options or has any specific preferences."
    }},
    {{
        "step": 4,
        "task": "Confirm if the user is ready to proceed with a purchase or needs more help."
    }},
    {{
        "step": 5,
        "task": "Persuade the user to sign up for the Prime membership."
    }}
]
Resources:
{{
    "MessageWorker": "The worker responsible for interacting with the user with predefined responses",
    "RAGWorker": "Answer the user's questions based on the company's internal documentations, such as the policies, FAQs, and product information",
    "ProductWorker": "Access the company's database to retrieve information about products, such as availability, pricing, and specifications",
    "UserProfileWorker": "Access the company's database to retrieve information about the user's preferences and history"
}}
Answer:
```json
[
    {{
        "step": 1,
        "task": "Retrieve the information about the customer from CRM and Inquire about specific preferences or requirements (e.g., brand, features, price range)."
        "resource": "UserProfileWorker",
        "example_response": "Do you have some specific preferences or requirements for the product you are looking for?"
    }},
    {{
        "step": 2,
        "task": "Provide a curated list of products that match the user's criteria."
        "resource": "ProductWorker",
        "example_response": ""
    }},
    {{
        "step": 3,
        "task": "Ask if the user would like to see more options or has any specific preferences."
        "resource": "MessageWorker",
        "example_response": "Would you like to see more options or do you have any specific preferences?"
    }},
    {{
        "step": 4,
        "task": "Confirm if the user is ready to proceed with a purchase or needs more help."
        "resource": "MessageWorker",
        "example_response": "Are you ready to proceed with the purchase or do you need more help?"
    }},
    {{
        "step": 5,
        "task": "Persuade the user to sign up for the Prime membership."
        "resource": "MessageWorker",
        "example_response": "I noticed that you are a frequent shopper. Have you considered signing up for our Prime membership to enjoy exclusive benefits and discounts?"
    }}
]
```

Best Practice: {best_practice}
Resources: {resources}
Answer:
"""

embed_reusable_task_resources_sys_prompt = """The builder plans to create an assistant designed to provide services to users. Given the best practices for addressing a specific task, and the available resources, your task is to map the steps with the resources. The response should include only the most suitable resource used for each step and example responses, if applicable. Return the answer in JSON format. Do not add any comment on the answer.
For example:
Best Practice: 
{{
    "task": "Retrieve the information about the customer from CRM and Inquire about specific preferences or requirements (e.g., brand, features, price range).",
    "next": [
        "task": "Provide a curated list of products that match the user's criteria.",
        "next": [
            "task": "Ask if the user would like to see more options or has any specific preferences.",
            "next": [
                "task": "Confirm if the user is ready to proceed with a purchase or needs more help.",
                "next": [
                    "task": "Persuade the user to sign up for the Prime membership.",
                    next: []
                ]
            ]
        ]
    ]
}}
Resources:
{{
    "MessageWorker": "The worker responsible for interacting with the user with predefined responses",
    "RAGWorker": "Answer the user's questions based on the company's internal documentations, such as the policies, FAQs, and product information",
    "ProductWorker": "Access the company's database to retrieve information about products, such as availability, pricing, and specifications",
    "UserProfileWorker": "Access the company's database to retrieve information about the user's preferences and history"
}}
Answer:
```json
{{
    "task": "Retrieve the information about the customer from CRM and Inquire about specific preferences or requirements (e.g., brand, features, price range).",
    "resource": "UserProfileWorker",
    "example_response": "Do you have some specific preferences or requirements for the product you are looking for?",
    "next": [
        "task": "Provide a curated list of products that match the user's criteria.",
        "resource": "ProductWorker",
        "example_response": "",
        "next": [
            "task": "Ask if the user would like to see more options or has any specific preferences.",
            "resource": "MessageWorker",
            "example_response": "Would you like to see more options or do you have any specific preferences?",
            "next": [
                "task": "Confirm if the user is ready to proceed with a purchase or needs more help.",
                "resource": "MessageWorker",
                "example_response": "Are you ready to proceed with the purchase or do you need more help?",
                "next": [
                    "task": "Persuade the user to sign up for the Prime membership.",
                    "resource": "MessageWorker",
                    "example_response": "I noticed that you are a frequent shopper. Have you considered signing up for our Prime membership to enjoy exclusive benefits and discounts?",
                    next: []
                ]
            ]
        ]
    ]
}}
```
Best Practice: {best_practice}
Resources: {resources}
Answer:
"""


generate_start_msg = """The builder plans to create a chatbot designed to fulfill user's objectives. Given the role of the chatbot, your task is to generate a starting message for the chatbot. Return the response in JSON format.

For Example:

Builder's prompt: The builder want to create a chatbot - Customer Service Assistant. The customer service assistant typically handles tasks such as answering customer inquiries, making product recommendations, assisting with orders, processing returns and exchanges, supporting billing and payments, addressing complaints, and managing customer accounts.
Start Message:
```json
{{
    "message": "Welcome to our Customer Service Assistant! How can I help you today?"
}}
```

Builder's prompt: The builder want to create a chatbot - {role}. {u_objective}
Start Message:
"""

task_intents_prediction_prompt = """The builder plans to create a chatbot designed to fulfill user's objectives. Given the role of the chatbot, along with any introductory information, detailed documentation (if available) and a list of tasks, your task is to identify the user's intent based on given tasks .Ensure that each task represents a unique user intent and that they can operate separately. Moreover, you are given the instructions that you must follow. Return the response in JSON format.

For Example:

Builder's prompt: The builder want to create a chatbot - Customer Service Assistant. The customer service assistant typically handles tasks such as answering customer inquiries, making product recommendations, assisting with orders, processing returns and exchanges, supporting billing and payments, addressing complaints, and managing customer accounts.
Builder's Information: Amazon.com is a large e-commerce platform that sells a wide variety of products, ranging from electronics to groceries.
Builder's documentations: 
https://www.amazon.com/
Holiday Deals
Disability Customer Support
Same-Day Delivery
Medical Care
Customer Service
Amazon Basics
Groceries
Prime
Buy Again
New Releases
Pharmacy
Shop By Interest
Amazon Home
Amazon Business
Subscribe & Save
Livestreams
luwanamazon's Amazon.com
Best Sellers
Household, Health & Baby Care
Sell
Gift Cards

https://www.amazon.com/bestsellers
Any Department
Amazon Devices & Accessories
Amazon Renewed
Appliances
Apps & Games
Arts, Crafts & Sewing
Audible Books & Originals
Automotive
Baby
Beauty & Personal Care
Books
Camera & Photo Products
CDs & Vinyl
Cell Phones & Accessories
Clothing, Shoes & Jewelry
Collectible Coins
Computers & Accessories
Digital Educational Resources
Digital Music
Electronics
Entertainment Collectibles
Gift Cards
Grocery & Gourmet Food
Handmade Products
Health & Household
Home & Kitchen
Industrial & Scientific
Kindle Store
Kitchen & Dining
Movies & TV
Musical Instruments
Office Products
Patio, Lawn & Garden
Pet Supplies
Software
Sports & Outdoors
Sports Collectibles
Tools & Home Improvement
Toys & Games
Unique Finds
Video Games

Tasks:
```json
[
    {{
        "task": "Provide help in Product Search and Discovery"
    }},
    {{
        "task": "Provide help in product inquiry"
    }},
    {{
        "task": "Provide help in product comparison"
    }},
    {{
        "task": "Provide help in billing and payment support"
    }},
    {{
        "task": "Provide help in order management"
    }},
    {{
        "task": "Provide help in Returns and Exchanges"
    }}
]
```


Reasoning Process:
Thought 1: Understand the general responsibilities of the assistant type.
Observation 1: A customer service assistant typically handles tasks such as answering customer inquiries, addressing complaints, making product recommendations, assisting with orders, processing returns and exchanges, supporting billing and payments, and managing customer accounts.

Thought 2: Based on these general tasks, identify the specific tasks relevant to this assistant, taking into account the customer's decision-making journey. Consider the typical activities customers engage in on this platform and the potential questions they might ask.
Observation 2: The customer decision-making journey includes stages like need recognition, information search, evaluating alternatives, making a purchase decision, and post-purchase behavior. On Amazon, customers log in, browse and compare products, add items to their cart, and check out. They also track orders, manage returns, and leave reviews. Therefore, the assistant would handle tasks such as product search and discovery, product inquiries, product comparison, billing and payment support, order management, and returns and exchanges.

Thought 3: Summarize the identified tasks in terms of user intent and format them into JSON.
Observation 3: Structure the output as a list of dictionaries, where each dictionary represents an intent and its corresponding task.

Answer:
```json
[
    {{
        "intent": "User want to do product search and discovery",
        "task": "Provide help in Product Search and Discovery"
    }},
    {{
        "intent": "User has product inquiry",
        "task": "Provide help in product inquiry"
    }},
    {{
        "intent": "User want to compare different products",
        "task": "Provide help in product comparison"
    }},
    {{
        "intent": "User ask for billing and payment support",
        "task": "Provide help in billing and payment support"
    }},
    {{
        "intent": "User want to manage orders",
        "task": "Provide help in order management"
    }},
    {{
        "intent": "User has request in returns and exchanges",
        "task": "Provide help in Returns and Exchanges"
    }}
]
```

Builder's prompt: The builder want to create a chatbot - {role}. {u_objective}
Builder's information: {intro}
Builder's documentations: 
{docs}
Builder's instructions: 
{instructions}
Tasks: 
{user_tasks}
Reasoning Process:
"""


class PromptManager:
    """Manages prompt templates for task generation."""

    def __init__(self) -> None:
        """Initialize the prompt manager."""
        self.generate_tasks_sys_prompt = generate_tasks_sys_prompt
        self.generate_reusable_tasks_sys_prompt = generate_reusable_tasks_sys_prompt
        self.check_best_practice_sys_prompt = check_best_practice_sys_prompt
        self.generate_best_practice_sys_prompt = generate_best_practice_sys_prompt
        self.embed_resources_sys_prompt = embed_resources_sys_prompt
        self.embed_builder_obj_sys_prompt = embed_builder_obj_sys_prompt
        self.task_intents_prediction_prompt = task_intents_prediction_prompt
        self.prompts = {
            "generate_tasks": generate_tasks_sys_prompt,
            "generate_reusable_tasks": generate_reusable_tasks_sys_prompt,
            "check_best_practice": check_best_practice_sys_prompt,
            "generate_intents": generate_intents_sys_prompt,
            "generate_best_practice": generate_best_practice_sys_prompt,
            "embed_resources": embed_resources_sys_prompt,
            "embed_builder_obj": embed_builder_obj_sys_prompt,
        }

    def get_prompt(self, name: str, **kwargs: object) -> str:
        """
        Get a formatted prompt by name.

        Args:
            name: The name of the prompt to get
            **kwargs: Formatting arguments for the prompt

        Returns:
            The formatted prompt string
        """
        prompt_template = self.prompts.get(name)
        if prompt_template is None:
            raise ValueError(f"Prompt template '{name}' not found.")
        return prompt_template.format(**kwargs)
