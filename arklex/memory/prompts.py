"""Prompt templates for memory management in the Arklex framework.

This module contains the prompt templates used by the memory system to personalize
user intents and manage conversation context. It includes prompts for intent
personalization, chain-of-thought reasoning, and example conversations that
demonstrate how to extract and format product and attribute information from
user queries.

Key Components:
- Introduction Prompt: Sets up the context and goal for intent personalization
- Chain-of-Thought Steps: Guides the reasoning process for intent extraction
- Example Conversations: Demonstrates proper intent personalization patterns
- Output Instructions: Defines the format and rules for intent generation

Key Features:
- Structured reasoning process for intent extraction
- Clear examples of different conversation scenarios
- Strict rules against attribute hallucination
- Consistent output formatting
- Context-aware intent personalization
"""

# Introduction prompt that sets up the context and goal for intent personalization
intro = """
You are helping personalize user intents in an agentic taskgraph workflow.
Your job: 
Given the tool's final raw output: tool_output, task performed by the tool: task, tool's context generated response: context_generate, basic user intent: user_intent, and current user utterance: user_utterance → output a detailed Chain-of-Thought reasoning followed by a short clean Personalized Intent focused on product + attribute.
"""

# Step-by-step instructions for the chain-of-thought reasoning process
cot_steps = """
INSTRUCTIONS (Chain-of-Thought):
Thought 1: From the tool output, identify each product card and its standout feature.
Thought 2: List bullets of key points for each product (design, material, inventory).
Thought 3: Formulate a concise conversational summary of all products. Double check the user_utterance to make sure that you are not adding extra attributes to the intent which they have not even mentioned.
Thought 4: Use that summary plus Task, Agent Response, Basic Intent, and User Utterance to derive a personalized intent focusing on the product and attribute. 
"""

# Example conversations demonstrating proper intent personalization patterns
final_examples = """
EXAMPLE 1 (General list → user picks one):

Output:
• Wireless Noise‑Cancelling Headphones – over‑ear, 30h battery, Bluetooth 5.0, inventory 120.
• Stainless Steel Travel Mug – 16oz, vacuum insulated, leak‑proof lid, inventory 200.
• Mesh Running Sneakers – breathable, size 6–12, cushioned sole, inventory 75.

Task: List available products
Context Generate: We offer a variety of gear and accessories. What are you interested in?
User Intent: User is browsing product categories
User Utterance: What can I choose from today?

Chain-of-Thought:
Thought 1: Three categories: headphones, travel mugs, and sneakers.
Thought 2:
- Headphones: long battery life and noise‑cancellation.
- Mug: keeps drinks hot/cold, leak‑proof.
- Sneakers: breathable mesh, cushioned sole.
Thought 3: Summarize and prompt choice. Double check the user_utterance to make sure that you are not inferring attributes which have NOT been mentioned by the user!
We have durable noise‑cancelling headphones, a stainless steel travel mug, and breathable running sneakers. Which would you like?
Thought 4: Derive intent based on this summary, task, context_generate, user_intent, user_utterance. From the user utterance and user intent, it is clear that the user is not talking about any specific product, they merely want to know the products in the store.

Personalized Intent: 
intent: User wants to know the different kinds of products at the store.
product: general list of store products  
attribute: none

EXAMPLE 2 (Follow‑up attribute request):

Output:
• Wireless Noise‑Cancelling Headphones – battery 30h, Bluetooth 5.0.
• Wired Studio Headphones – coiled cable, 50mm drivers.

Task: List available products
Context Generate: We have many headphones. Do you have a preference?
User Intent: User wants information about products
User Utterance: Do you have headphones?

Chain-of-Thought:
Thought 1: Identify the product cards to be of headphones.
Thought 2: List attributes
- Wireless: 30h battery.
- Wired: no battery required.
Thought 3: Summarize: We have two headphones, one being wireless lasts 30h, and the other—wired—draws from device. Double check the user_utterance to make sure that you are not inferring attributes which have NOT been mentioned by the user!
Thought 4: Derive intent based on this summary, task, context_generate, user_intent, user_utterance. Think about it. Don't infer product or attributes that the user has not mentioned. The user is asking specifically for headphones, and we have two models of them to offer.

Personalized Intent: 
intent: User wants to know if we have headphones in stock.
product: headphones  
attribute: none

EXAMPLE 3 (Attribute‑specific follow‑up):

Output:
• Mesh Running Sneakers – available sizes 6–12, cushioned sole.
• Trail Running Sneakers – waterproof, grippy outsole.

Task: Recommend products
Context-Generate: We have running sneakers in stock. Interested in a specific feature?
User Intent: User wants recommendation
User Utterance: Do you have waterproof options?

Chain-of-Thought:
Thought 1: Identify sneakers as the product.
Thought 2: Note the features of both—mesh, trail, available sizes, waterproof, grippy outsole.
Thought 3: Summarize that choice. Double check the user_utterance to make sure that you are not inferring attributes which have NOT been mentioned by the user!
Thought 4: Derive intent based on this summary, task, context_generate, user_intent, user_utterance. Think about it. Don't infer product or attributes that the user has not mentioned. User now seeking sneaker recommendations but in waterproof category.

Personalized Intent: 
intent: User is looking for waterproof sneakers.
product: sneakers  
attribute: waterproof
"""

# Instructions for generating the final personalized intent
output_instructions = """
**Important Cases:**
1. If user mentions a product, identify the product and also the attribute, IF ANY.
2. If user mentions only attribute, infer most likely product based on context and based on the given information.
3. If there are multiple products mentioned, identify all of them and infer their individual attributes, IF ANY.
4. If user asks a follow-up, combine previous context.

DO NOT HALLUCINATE ATTRIBUTES OR PRODUCTS THAT THE USER HAS NOT MENTIONED. This is very important! Only mention the attributes that the user has mentioned in user_utterance!!
Make sure to only infer the products and attributes that the user has mentioned, do not hallucinate based on only the summarized output. Sometimes the attributes may not exist, in which case don't display it in the personalized intent.

**Output Format:**
Write the final personalized intent as a short clean phrase, followed by the product and attribute inferred  (IF ANY), by your reasoning.
Do not add explanations or extra text.

Now generate the final personalized intent for the following input information:
"""

# Reference examples for product/attribute personalization (commented out)
"""
pa_examples = '''
EXAMPLE 1 (General list → user picks one):

Output:
• Wireless Noise‑Cancelling Headphones – over‑ear, 30h battery, Bluetooth 5.0, inventory 120.
• Stainless Steel Travel Mug – 16oz, vacuum insulated, leak‑proof lid, inventory 200.
• Mesh Running Sneakers – breathable, size 6–12, cushioned sole, inventory 75.

Task: List available products
Context Generate: We offer a variety of gear and accessories. What are you interested in?
User Intent: User is browsing product categories
User Utterance: What can I choose from today?

Chain-of-Thought:
Thought 1: Three categories: headphones, travel mugs, and sneakers.
Thought 2:
- Headphones: long battery life and noise‑cancellation.
- Mug: keeps drinks hot/cold, leak‑proof.
- Sneakers: breathable mesh, cushioned sole.
Thought 3: Summarize and prompt choice. Double check the user_utterance to make sure that you are not inferring attributes which have NOT been mentioned by the user!
We have durable noise‑cancelling headphones, a stainless steel travel mug, and breathable running sneakers. Which would you like?
Thought 4: Derive intent based on this summary, task, context_generate, user_intent, user_utterance. From the user utterance and user intent, it is clear that the user is not talking about any specific product, they merely want to know the products in the store.

Personalized Intent: 
intent: User wants to know the different kinds of products at the store.
product: general list of store products  
attribute: none

EXAMPLE 2 (Follow‑up attribute request):

Output:
• Wireless Noise‑Cancelling Headphones – battery 30h, Bluetooth 5.0.
• Wired Studio Headphones – coiled cable, 50mm drivers.

Task: List available products
Context Generate: We have many headphones. Do you have a preference?
User Intent: User wants information about products
User Utterance: Do you have headphones?

Chain-of-Thought:
Thought 1: Identify the product cards to be of headphones.
Thought 2: List attributes
- Wireless: 30h battery.
- Wired: no battery required.
Thought 3: Summarize: We have two headphones, one being wireless lasts 30h, and the other—wired—draws from device. Double check the user_utterance to make sure that you are not inferring attributes which have NOT been mentioned by the user!
Thought 4: Derive intent based on this summary, task, context_generate, user_intent, user_utterance. Think about it. Don't infer product or attributes that the user has not mentioned. The user is asking specifically for headphones, and we have two models of them to offer.

Personalized Intent: 
intent: User wants to know if we have headphones in stock.
product: headphones  
attribute: none

EXAMPLE 3 (Attribute‑specific follow‑up):

Output:
• Mesh Running Sneakers – available sizes 6–12, cushioned sole.
• Trail Running Sneakers – waterproof, grippy outsole.

Task: Recommend products
Context-Generate: We have running sneakers in stock. Interested in a specific feature?
User Intent: User wants recommendation
User Utterance: Do you have waterproof options?

Chain-of-Thought:
Thought 1: Identify sneakers as the product.
Thought 2: Note the features of both—mesh, trail, available sizes, waterproof, grippy outsole.
Thought 3: Summarize that choice. Double check the user_utterance to make sure that you are not inferring attributes which have NOT been mentioned by the user!
Thought 4: Derive intent based on this summary, task, context_generate, user_intent, user_utterance. Think about it. Don't infer product or attributes that the user has not mentioned. User now seeking sneaker recommendations but in waterproof category.

Personalized Intent: 
intent: User is looking for waterproof sneakers.
product: sneakers  
attribute: waterproof
'''
"""

# Reference examples for intent-only personalization (commented out)
"""
intent_examples = '''
EXAMPLE 1 (General list → user picks one):

Output:
• Wireless Noise‑Cancelling Headphones – over‑ear, 30h battery, Bluetooth 5.0, inventory 120.
• Stainless Steel Travel Mug – 16oz, vacuum insulated, leak‑proof lid, inventory 200.
• Mesh Running Sneakers – breathable, size 6–12, cushioned sole, inventory 75.

Task: List available products
Context Generate: We offer a variety of gear and accessories. What are you interested in?
User Intent: User is browsing product categories
User Utterance: What can I choose from today?

Chain-of-Thought:
Thought 1: Three categories: headphones, travel mugs, and sneakers.
Thought 2:
- Headphones: long battery life and noise‑cancellation.
- Mug: keeps drinks hot/cold, leak‑proof.
- Sneakers: breathable mesh, cushioned sole.
Thought 3: Summarize and prompt choice. Double check the user_utterance to make sure that you are not inferring attributes which have NOT been mentioned by the user!
We have durable noise‑cancelling headphones, a stainless steel travel mug, and breathable running sneakers. Which would you like?
Thought 4: Derive intent based on this summary, task, context_generate, user_intent, user_utterance. From the user utterance and user intent, it is clear that the user is not talking about any specific product, they merely want to know the products in the store.

Personalized Intent: 
intent: User wants to know the different kinds of products at the store.
product: general list of store products  
attribute: none

EXAMPLE 2 (Follow‑up attribute request):

Output:
• Wireless Noise‑Cancelling Headphones – battery 30h, Bluetooth 5.0.
• Wired Studio Headphones – coiled cable, 50mm drivers.

Task: List available products
Context Generate: We have many headphones. Do you have a preference?
User Intent: User wants information about products
User Utterance: Do you have headphones?

Chain-of-Thought:
Thought 1: Identify the product cards to be of headphones.
Thought 2: List attributes
- Wireless: 30h battery.
- Wired: no battery required.
Thought 3: Summarize: We have two headphones, one being wireless lasts 30h, and the other—wired—draws from device. Double check the user_utterance to make sure that you are not inferring attributes which have NOT been mentioned by the user!
Thought 4: Derive intent based on this summary, task, context_generate, user_intent, user_utterance. Think about it. Don't infer product or attributes that the user has not mentioned. The user is asking specifically for headphones, and we have two models of them to offer.

Personalized Intent: 
intent: User wants to know if we have headphones in stock.
product: headphones  
attribute: none

EXAMPLE 3 (Attribute‑specific follow‑up):

Output:
• Mesh Running Sneakers – available sizes 6–12, cushioned sole.
• Trail Running Sneakers – waterproof, grippy outsole.

Task: Recommend products
Context-Generate: We have running sneakers in stock. Interested in a specific feature?
User Intent: User wants recommendation
User Utterance: Do you have waterproof options?

Chain-of-Thought:
Thought 1: Identify sneakers as the product.
Thought 2: Note the features of both—mesh, trail, available sizes, waterproof, grippy outsole.
Thought 3: Summarize that choice. Double check the user_utterance to make sure that you are not inferring attributes which have NOT been mentioned by the user!
Thought 4: Derive intent based on this summary, task, context_generate, user_intent, user_utterance. Think about it. Don't infer product or attributes that the user has not mentioned. User now seeking sneaker recommendations but in waterproof category.

Personalized Intent: 
intent: User is looking for waterproof sneakers.
product: sneakers  
attribute: waterproof
'''
"""
