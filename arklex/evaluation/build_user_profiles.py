"""User profile generation and management for evaluation in the Arklex framework.

This module provides functionality for building synthetic user profiles for evaluation
purposes. It includes tools for generating user attributes, adapting goals to specific
company contexts, and managing system attributes with binding support.
"""

import random
from typing import Any, Literal

import anthropic
import requests
from openai import OpenAI

from arklex.env.env import Environment
from arklex.env.tools.tools import Tool
from arklex.evaluation.chatgpt_utils import chatgpt_chatbot
from arklex.evaluation.get_documents import load_docs
from arklex.orchestrator.NLU.core.slot import SlotFiller
from arklex.utils.logging_utils import LogContext

# Type aliases for better readability
StrategyType = Literal["react", "llm_based", "random"]
DocumentType = dict[str, str]  # {"content": str, ...}
AttributeDict = dict[str, Any]
SystemAttributeDict = dict[str, Any]
ConfigDict = dict[str, Any]
GoalList = list[str]
ProfileList = list[str]
LabelDict = dict[str, Any]

ATTR_TO_PROFILE: str = "Convert the following list user attributes in to a text description of a customer profile for the following company:\n{company_summary}\nThe user attributes are here:\n{user_attr}"
ADAPT_GOAL: str = "Assume you are planning to speak to a chatbot with the following goal in mind:\n{goal}\nUsing the company information below, re-write this goal into one that is more specific to the company and align with your profile. The new goal should be more specific either relevent to your profile or the company's details. Here is a summary of the company:\n{company_summary}\n{doc}\n{user_profile}"
ADD_ATTRIBUTES: str = "Your job is to add attributes to a customer profile. Here is an example of an existing profile with the categories on the left and the attributes on the right:\n{user_profile}\nSuggest three attributes for the following category:\n{category}\nThese attributes should be specific values that are relevant to the category and apply to potential customers of the company. You should return a comma separated list of attributes without any descriptions of the attributes. Generated the attributes based on a summary of the company and the company webpage and what kind of customers the company is likely targeting. Here is the summary of the company:\n{company_summary}\nHere is the webpage:\n{company_doc}"
ADD_ATTRIBUTES_WO_DOC: str = "Your job is to add attributes to a customer profile. Here is an example of an existing profile with the categories on the left and the attributes on the right:\n{user_profile}\nSuggest three attributes for the following category:\n{category}\nThese attributes should be specific values that are relevant to the category and apply to potential customers of the company. You should return a comma separated list of attributes without any descriptions of the attributes. Generated the attributes based on a summary of the company and what kind of customers the company is likely targeting. Here is the summary of the company:\n{company_summary}"

log_context: LogContext = LogContext(__name__)


def _get_random_document_content(documents: list[DocumentType]) -> str:
    """Get random document content or empty string if no documents exist.

    Args:
        documents: List of document dictionaries with 'content' keys.

    Returns:
        Formatted document content string prefixed with "Here is a page from the company website: "
        or empty string if no documents are available.
    """
    return (
        "Here is a page from the company website: "
        + random.choice(documents)["content"]
        if documents and len(documents) > 0
        else ""
    )


def _format_user_profile(attributes: AttributeDict) -> str:
    """Format user attributes into a profile string.

    Args:
        attributes: Dictionary of user attributes with key-value pairs.

    Returns:
        Formatted profile string prefixed with "Here is the your profile: " followed by
        semicolon-separated key-value pairs.
    """
    return "Here is the your profile: " + "; ".join(
        f"{key}: {value}" for key, value in attributes.items()
    )


def _process_attributes_for_conversation(
    user_profile: AttributeDict,
    augmented_attributes: dict[str, list[str]],
    config: ConfigDict,
    documents: list[DocumentType],
) -> tuple[AttributeDict, LabelDict]:
    """Process attributes for a single conversation with goal adaptation.

    This function takes user profile attributes, augments them with additional values,
    picks appropriate attributes based on goals, and adapts the goal to be more specific
    to the company context.

    Args:
        user_profile: User profile dictionary containing current attributes.
        augmented_attributes: Dictionary mapping attribute categories to lists of possible values.
        config: Configuration dictionary containing client, user_attributes, and company_summary.
        documents: List of document dictionaries with company website content.

    Returns:
        Tuple containing:
        - Processed attributes dictionary with adapted goal
        - Matched attribute to goal mapping for labeling
    """
    strategy: StrategyType = "react"
    attributes, matched_attribute_to_goal = pick_attributes(
        user_profile,
        augmented_attributes,
        config["user_attributes"]["goal"]["values"],
        strategy=strategy,
        client=config["client"],
    )

    doc: str = _get_random_document_content(documents)
    user_profile_str: str = _format_user_profile(attributes)
    goal: str = adapt_goal(
        goal=attributes["goal"],
        config=config,
        doc=doc,
        user_profile=user_profile_str,
    )
    attributes["goal"] = goal
    return attributes, matched_attribute_to_goal


def _select_random_from_list(data_list: list[Any], key: str) -> tuple[Any | None, int]:
    """Select random item from list and return item with index.

    Args:
        data_list: List to select from.
        key: Key for logging purposes when list is empty.

    Returns:
        Tuple of selected item (or None if empty) and index (or -1 if empty).
    """
    if not data_list:
        log_context.warning(f"Empty list for {key}")
        return None, -1
    random_index: int = random.choice(range(len(data_list)))
    return data_list[random_index], random_index


def _process_system_attributes(
    config: ConfigDict, system_attributes: SystemAttributeDict
) -> tuple[SystemAttributeDict, dict[str, int]]:
    """Process system attributes and create binding index for coordinated selection.

    This function handles system attributes that may be bound to other attributes,
    ensuring coordinated selection across related attribute categories.

    Args:
        config: Configuration dictionary containing user_attributes.system_attributes.
        system_attributes: Dictionary of system attributes with lists of possible values.

    Returns:
        Tuple containing:
        - Processed system attributes dictionary with selected values
        - Binding index mapping bound attribute names to their selected indices
    """
    system_attribute: SystemAttributeDict = {}
    binding_index: dict[str, int] = {}

    for key, value in config["user_attributes"]["system_attributes"].items():
        if "bind_to" in value:
            selected_item, random_index = _select_random_from_list(
                system_attributes[key], key
            )
            system_attribute[key] = selected_item
            if random_index >= 0:
                binding_index[value["bind_to"]] = random_index
        else:
            selected_item, _ = _select_random_from_list(system_attributes[key], key)
            system_attribute[key] = selected_item

    return system_attribute, binding_index


def _process_user_profiles(
    config: ConfigDict, user_profiles: dict[str, Any], binding_index: dict[str, int]
) -> AttributeDict:
    """Process user profiles with binding support for coordinated attribute selection.

    This function selects user profile attributes, respecting any binding relationships
    established in the system attributes processing.

    Args:
        config: Configuration dictionary containing user_attributes.user_profiles.
        user_profiles: Dictionary of user profiles with lists of possible values.
        binding_index: Binding index mapping bound attribute names to their selected indices.

    Returns:
        Processed user profile dictionary with selected values respecting bindings.
    """
    user_profile: AttributeDict = {}

    for key, value in config["user_attributes"]["user_profiles"].items():
        if "bind_to" in value and value["bind_to"] in binding_index:
            if user_profiles[key]:
                user_profile[key] = user_profiles[key][binding_index[value["bind_to"]]]
            else:
                log_context.warning(f"Empty user profiles list for {key}")
                user_profile[key] = None
        elif "bind_to" not in value:
            selected_item, _ = _select_random_from_list(user_profiles[key], key)
            user_profile[key] = selected_item

    return user_profile


def build_profile(
    synthetic_data_params: dict[str, Any], config: ConfigDict
) -> tuple[
    ProfileList,
    GoalList,
    list[AttributeDict],
    list[SystemAttributeDict],
    list[LabelDict],
]:
    """Build user profiles for evaluation with synthetic data generation.

    This is the main function for generating user profiles for evaluation purposes.
    It can work in two modes: standard mode using predefined attributes, or custom
    mode using external API data with binding support.

    Args:
        synthetic_data_params: Parameters for synthetic data generation including
            num_convos (int): Number of conversations to generate
            num_goals (int): Number of goals to consider
        config: Configuration settings for profile generation including
            documents_dir (str): Directory containing company documents
            custom_profile (bool): Whether to use custom profiles from API
            system_inputs (bool): Whether to include system inputs
            client: LLM client for goal adaptation
            company_summary (str): Summary of the company
            user_attributes (dict): Configuration for user attributes
            tools (list): Available tools for the assistant
            workers (list): Available workers for the assistant

    Returns:
        Tuple containing:
        - List of profile descriptions (str)
        - List of goals (str)
        - List of attribute dictionaries
        - List of system input dictionaries
        - List of label dictionaries for evaluation
    """
    labels_list: list[LabelDict] = []
    attributes_list: list[AttributeDict] = []
    system_attributes_list: list[SystemAttributeDict] = []

    documents = load_docs(
        config["documents_dir"], config, synthetic_data_params["num_goals"] * 2
    )
    predefined_attributes = filter_attributes(config)
    augmented_attributes = augment_attributes(predefined_attributes, config, documents)

    if not config["custom_profile"]:
        user_profile = {}
        for _ in range(synthetic_data_params["num_convos"]):
            attributes, matched_attribute_to_goal = (
                _process_attributes_for_conversation(
                    user_profile, augmented_attributes, config, documents
                )
            )
            labels_list.append(matched_attribute_to_goal)
            attributes_list.append(attributes)

        system_attributes_list = (
            select_system_attributes(config, synthetic_data_params)
            if config["system_inputs"]
            else [{}] * synthetic_data_params["num_convos"]
        )
    else:
        user_profiles, system_attributes = get_custom_profiles(config)
        for _ in range(synthetic_data_params["num_convos"]):
            system_attribute, binding_index = _process_system_attributes(
                config, system_attributes
            )
            user_profile = _process_user_profiles(config, user_profiles, binding_index)

            attributes, matched_attribute_to_goal = (
                _process_attributes_for_conversation(
                    user_profile, augmented_attributes, config, documents
                )
            )
            labels_list.append(matched_attribute_to_goal)
            attributes_list.append(attributes)
            system_attributes_list.append(system_attribute)

    profiles, goals, system_inputs = convert_attributes_to_profiles(
        attributes_list, system_attributes_list, config
    )
    return profiles, goals, attributes_list, system_inputs, labels_list


def pick_goal(
    attributes: AttributeDict,
    goals: GoalList,
    strategy: StrategyType = "react",
    client: OpenAI | anthropic.Anthropic | None = None,
) -> str:
    """Pick a goal based on user attributes and specified strategy.

    This function selects the most appropriate goal from a list of available goals
    based on the user's attributes. It supports different strategies including
    LLM-based reasoning and reactive selection.

    Args:
        attributes: Dictionary containing user attributes to consider for goal selection.
        goals: List of available goals to choose from.
        strategy: Strategy for goal selection. Options:
            - "llm_based": Use LLM to reason about the best goal
            - "react": Use LLM with reasoning chain (thought process)
        client: LLM client for goal selection (required for LLM-based strategies).

    Returns:
        Selected goal string.

    Raises:
        ValueError: If an invalid strategy is provided.
    """
    if strategy == "llm_based":
        PICK_GOAL_PROMPT = """Given the following user's attributes, please pick the most relevant goal from the given list of goals.
user's attributes:
{attributes}

Goals:
{goals}

Goal:
"""
        response = chatgpt_chatbot(
            [
                {
                    "role": "user",
                    "content": PICK_GOAL_PROMPT.format(
                        goals="\n".join(goals), attributes=attributes
                    ),
                }
            ],
            client=client,
        )
        goal = response.split("Goal:")[1].strip()
        print("goal: ", goal)
    elif strategy == "react":
        PICK_GOAL_PROMPT = """Given the following user's attributes, please pick the most relevant goal from the given list of goals. First, generate a Thought about the reason why you pick this goal. Then, generate the final decided one attribute.
user's attributes:
{attributes}

Goals:
{goals}

Format:

Thought:
<the thought>

Goal:
<the picked goal>
"""
        response = chatgpt_chatbot(
            [
                {
                    "role": "user",
                    "content": PICK_GOAL_PROMPT.format(
                        goals="\n".join(goals), attributes=attributes
                    ),
                }
            ],
            client=client,
        )
        thought = response.split("Thought:")[1].split("Goal:")[0].strip()
        print("thought: ", thought)
        goal = response.split("Goal:")[1].strip()
        print("goal: ", goal)
    else:
        raise ValueError("Invalid strategy")
    return goal


def find_matched_attribute(
    goal: str,
    user_profile_or_attributes: str | AttributeDict,
    strategy: StrategyType = "react",
    client: OpenAI | anthropic.Anthropic | None = None,
) -> str | LabelDict:
    """Find the matched attribute for a given goal using specified strategy.

    This function identifies the most relevant attribute that should be provided
    to an assistant to achieve the given goal. It can work with either a user
    profile string or an attributes dictionary.

    Args:
        goal: The goal that needs to be achieved.
        user_profile_or_attributes: Either a formatted user profile string or
            a dictionary of user attributes.
        strategy: Strategy for attribute matching. Currently supports "react"
            which uses LLM reasoning to find the most relevant attribute.
        client: LLM client for attribute matching (required for LLM-based strategies).

    Returns:
        Either a string attribute value (for string input) or a dictionary
        containing goal and matched attributes (for dictionary input).

    Raises:
        ValueError: If an invalid strategy is provided.
    """
    if isinstance(user_profile_or_attributes, dict):
        matched_attribute_to_goal = {
            "goal": goal,
            "matched_attribute": user_profile_or_attributes,
        }
        return matched_attribute_to_goal

    user_profile_str = user_profile_or_attributes

    if strategy == "react":
        FIND_MATCHED_ATTRIBUTE_PROMPT = """Given the following goal, please find the most relevant attribute category and its corresponding values(it can come from the user's information, product information or other user's persona) from the full attributes that user need to provide to the assistant in order to let the assistant achieve the goal. First, generate a thought about the reason why you pick this attribute category and its corresponding values. Then, generate the final decided one attribute value. Please only return single attribute value.
For example, 
########################################################
1. 
Goal: interested in product information
Full attributes:
user_info: {{'id': 'gid://shopify/Customer/8740759797990', 'firstName': 'Yunan', 'lastName': 'Lu', 'email': 'yl4021@columbia.edu', 'phone': None, 'createdAt': '2025-03-23T02:47:38Z', 'updatedAt': '2025-03-29T21:01:02Z', 'numberOfOrders': '0', 'orders': {{'edges': []}}, 'amountSpent': {{'amount': '0.0', 'currencyCode': 'USD'}}, 'lastOrder': None, 'addresses': []}}
current_webpage: Product ID: gid://shopify/Product/8970006790374
Title: Pink Unicorn Boys & Girls Baseball Hat with Adjustable Buckle (One Size Fits Most)
Description: 𝐄𝐘𝐄-𝐂𝐀𝐓𝐂𝐇𝐈𝐍𝐆 – The Awhale Girl's Unicorn Baseball Hat stands out with a 3D design and graphics packed with a vibrant pink color and tons of personality. Your kid will not want to take it off! Add some magic to your child's wardrobe with this adorable baseball cap! 𝐏𝐄𝐑𝐅𝐄𝐂𝐓 𝐅𝐈𝐓 – Made for all girl's hair types, our hat contains 6 embroidered eyelets and a full back opening for those messy buns and ponytails. Designed to fit children ages 2-12, the adjustable buckle can be tweaked in seconds for toddlers or tweens! 𝐇𝐈𝐆𝐇-𝐐𝐔𝐀𝐋𝐈𝐓𝐘 – Made with Premium cotton, our girl's unicorn baseball hat stays stunning with machine-washable cotton twill and durable stitching that preserves the colors and personality of the hat. 𝐀𝐋𝐋-𝐃𝐀𝐘 𝐔𝐒𝐄 – Made with breathable material, our unicorn baseball hat is comfortable for outdoor activities like running, baseball, tennis, and golf but also perfect for casual wear at school, the park, or on a playdate! 𝐀𝐖𝐇𝐀𝐋𝐄 𝐁𝐀𝐍𝐃 – Welcome to AWHALE, where our designers are obsessed with combining High-Quality Materials and Chic Design to bring joy and laughter to boys and girls. Your child will love wearing our stylish outfits, and as everyone knows, there is nothing more adorable than a happy and fashionable child!
Total Inventory: 546
Options: [{{'name': 'Title', 'values': ['Default Title']}}]
The following are several variants of the product:
Variant name: Pink Unicorn Boys & Girls Baseball Hat with Adjustable Buckle (One Size Fits Most) - Default Title, Variant ID: gid://shopify/ProductVariant/45802049208550, Price: 19.99, Inventory Quantity: 546

product_experience_level: new to this product
customer_type: new prospect
persona: curious
current_webpage: about page
modality: text
communication_type: incoming
discovery_type: search engine results
buying_behavior: information gathering
budget: budget: low to moderate
Location: USA

Thought:
The user is interested in product information that they are looking at, so they probably have some question regarding the product's attribute, such as color, size, material, etc. In this case, the attribute category should be "product attribute" and the corresponding value can be color. 

Attribute:
product attribute: color

########################################################
2. 
Goal: return order
Full attributes:
user_info: {{'id': 'gid://shopify/Customer/8746986963174', 'firstName': 'two-orders', 'lastName': 'test-customer', 'email': 'two-orders-test@example.com', 'phone': None, 'createdAt': '2025-03-26T18:59:41Z', 'updatedAt': '2025-03-26T19:01:13Z', 'numberOfOrders': '2', 'orders': {{'edges': [{{'node': {{'id': 'gid://shopify/Order/6284126519526', 'name': '#1006', 'createdAt': '2025-03-26T19:00:09Z', 'cancelledAt': None, 'returnStatus': 'NO_RETURN', 'statusPageUrl': 'https://arklex-test-store.myshopify.com/73279963366/orders/7f635998c026a631847d1b5c68424234/authenticate?key=b63ae9312d8398e9b24df7b2b36aad4a', 'totalPriceSet': {{'presentmentMoney': {{'amount': '41.99'}}}}, 'fulfillments': [], 'lineItems': {{'edges': [{{'node': {{'id': 'gid://shopify/LineItem/15440574218470', 'title': 'Winter Flannel Blanket Solid Color Plaid Coral Blanket Fleece Bedspread For Bed Sofa Thicken Plush Blanket Thin Quilt Home Decor', 'quantity': 1, 'variant': {{'id': 'gid://shopify/ProductVariant/45802067525862', 'product': {{'id': 'gid://shopify/Product/8970009215206'}}}}}}}}]}}}}, {{'node': {{'id': 'gid://shopify/Order/6284127568102', 'name': '#1007', 'createdAt': '2025-03-26T19:01:12Z', 'cancelledAt': None, 'returnStatus': 'NO_RETURN', 'statusPageUrl': 'https://arklex-test-store.myshopify.com/73279963366/orders/6c2c4ee90b1befab9468978cbc1beb22/authenticate?key=510a7866400cfe4056f81a678ce9fdd9', 'totalPriceSet': {{'presentmentMoney': {{'amount': '16.99'}}}}, 'fulfillments': [], 'lineItems': {{'edges': [{{'node': {{'id': 'gid://shopify/LineItem/15440577298662', 'title': 'Inyahome New Art Velvet Yellow Blue Pink Solid Color Cushion Cover Pillow Cover Pillow Case Home Decorative Sofa Throw Decor', 'quantity': 1, 'variant': {{'id': 'gid://shopify/ProductVariant/45802063134950', 'product': {{'id': 'gid://shopify/Product/8970008461542'}}}}}}}}]}}}}}}]}}, 'amountSpent': {{'amount': '58.98', 'currencyCode': 'USD'}}, 'lastOrder': {{'id': 'gid://shopify/Order/6284127568102', 'name': '#1007'}}, 'addresses': [{{'id': 'gid://shopify/MailingAddress/9852296495334?model_name=CustomerAddress', 'firstName': 'two-orders', 'lastName': 'test-customer', 'company': '', 'address1': '2381 Dongan Pl', 'address2': '', 'city': 'New York', 'province': 'New York', 'country': 'United States', 'zip': '10040', 'phone': '+19999999999', 'name': 'two-orders test-customer', 'provinceCode': 'NY', 'countryCodeV2': 'US'}}]}}
current_webpage: Product ID: gid://shopify/Product/8970006855910
Title: White Rainbow Boys & Girls Baseball Hat with Adjustable Buckle(One Size Fits Most)
Description: 𝐄𝐘𝐄-𝐂𝐀𝐓𝐂𝐇𝐈𝐍𝐆 – The Awhale Girl's Unicorn Baseball Hat stands out with a 3D design and graphics packed with vibrant colors and tons of personality. Your kid will not want to take it off! Add some magic to your child's wardrobe with this adorable baseball cap! 𝐏𝐄𝐑𝐅𝐄𝐂𝐓 𝐅𝐈𝐓 – Made for all girl's hair types, our hat contains 6 embroidered eyelets and a full back opening for those messy buns and ponytails. Designed to fit children ages 2-12, the adjustable buckle can be tweaked in seconds for toddlers or tweens! 𝐇𝐈𝐆𝐇-𝐐𝐔𝐀𝐋𝐈𝐓𝐘 – Made with Premium cotton, our girl's unicorn baseball hat stays stunning with machine-washable cotton twill and durable stitching that preserves the colors and personality of the hat. 𝐀𝐋𝐋-𝐃𝐀𝐘 𝐔𝐒𝐄 – Made with breathable material, our unicorn baseball hat is comfortable for outdoor activities like running, baseball, tennis, and golf but also perfect for casual wear at school, the park, or on a playdate! 𝐀𝐖𝐇𝐀𝐋𝐄 𝐁𝐑𝐀𝐍𝐃 – Welcome to AWHALE, where our designers are obsessed with combining High-Quality Materials and Chic Design to bring joy and laughter to boys and girls. Your child will love wearing our stylish outfits, and as everyone knows, there is nothing more adorable than a happy and fashionable child!
Total Inventory: 499
Options: [{{'name': 'Title', 'values': ['Default Title']}}]
The following are several variants of the product:
Variant name: White Rainbow Boys & Girls Baseball Hat with Adjustable Buckle(One Size Fits Most) - Default Title, Variant ID: gid://shopify/ProductVariant/45802049372390, Price: 19.99, Inventory Quantity: 499

product_experience_level: new to this product
customer_type: returning customer
persona: neutral
current_webpage: product page
modality: browsing
communication_type: responsive
discovery_type: search engine results
buying_behavior: value-conscious
budget: value-conscious budget
purchase_history: home_decor_enthusiast
Location: New York City, NY, USA

Thought:
The user has placed two orders, so they are likely to return one of the orders. In order to do so, user need to provide the order id that they want to return.

Attribute:
Order id: gid://shopify/Order/6284126519526

########################################################
3. Goal: order tracking
Full attributes:
user_info: {{'id': 'gid://shopify/Customer/8728033657062', 'firstName': 'Xinyang', 'lastName': 'Wang', 'email': 'xinyang.wang@arklex.ai', 'phone': None, 'createdAt': '2025-03-19T16:02:24Z', 'updatedAt': '2025-04-11T15:29:35Z', 'numberOfOrders': '2', 'orders': {{'edges': [{{'node': {{'id': 'gid://shopify/Order/6294747119846', 'name': '#1014', 'createdAt': '2025-04-03T19:37:43Z', 'cancelledAt': None, 'returnStatus': 'NO_RETURN', 'statusPageUrl': 'https://arklex-test-store.myshopify.com/73279963366/orders/0b6fb2edceb8b38625db4cd4041d45a2/authenticate?key=e6a64953a5636a37733887a77a4835d2', 'totalPriceSet': {{'presentmentMoney': {{'amount': '31.99'}}}}, 'fulfillments': [], 'lineItems': {{'edges': [{{'node': {{'id': 'gid://shopify/LineItem/15461470961894', 'title': 'Bedding Set Solid Color Luxury Bedding Kit Rayon Satin Duvet Cover Set Twin Queen King Size Bed Set 2pcs/3pcs/4pcs', 'quantity': 1, 'variant': {{'id': 'gid://shopify/ProductVariant/45802057138406', 'product': {{'id': 'gid://shopify/Product/8970007970022'}}}}}}}}]}}}}, {{'node': {{'id': 'gid://shopify/Order/6294747807974', 'name': '#1015', 'createdAt': '2025-04-03T19:38:16Z', 'cancelledAt': '2025-04-03T19:40:33Z', 'returnStatus': 'NO_RETURN', 'statusPageUrl': 'https://arklex-test-store.myshopify.com/73279963366/orders/d76cae23bdc06689d3d7f4955978c966/authenticate?key=289ab7019d0e6ad3a0474e678618180b', 'totalPriceSet': {{'presentmentMoney': {{'amount': '15.99'}}}}, 'fulfillments': [], 'lineItems': {{'edges': [{{'node': {{'id': 'gid://shopify/LineItem/15461472436454', 'title': 'Green Boys & Girls Baseball Hat with Adjustable Buckle', 'quantity': 1, 'variant': {{'id': 'gid://shopify/ProductVariant/45802048487654', 'product': {{'id': 'gid://shopify/Product/8970006659302'}}}}}}}}]}}}}}}]}}, 'amountSpent': {{'amount': '31.99', 'currencyCode': 'USD'}}, 'lastOrder': {{'id': 'gid://shopify/Order/6294747807974', 'name': '#1015'}}, 'addresses': [{{'id': 'gid://shopify/MailingAddress/9835887526118?model_name=CustomerAddress', 'firstName': 'Xinyang', 'lastName': 'Wang', 'company': None, 'address1': '515 West 113th Street', 'address2': None, 'city': 'New York', 'province': 'New York', 'country': 'United States', 'zip': '10025', 'phone': None, 'name': 'Xinyang Wang', 'provinceCode': 'NY', 'countryCodeV2': 'US'}}]}}
current_webpage: Product ID: gid://shopify/Product/8970008953062
Title: Flower Plush Throw Pillow Soft Plant Cartoon Chair Cushion Living Bedroom Home Decorative Pillows Sofa Cushions Birthday Gifts
Description: Origin: CN(Origin)Type: Seat Cushion/Back CushionFeature: MemorySet Type: NoUnpick and Wash: Not Removable and WashablePattern: PRINTEDis_customized: NoStyle: MEDITERRANEANModel Number: P161Technics: KnittedShape: RoundPattern Type: cartoonFilling: CottonMaterial: Polyester / CottonAge Group: AdultsDimensions: 32-35cm/42-45cm/52-55cmWarning: 3 years and up
Total Inventory: 0
Options: [{{'name': 'Color', 'values': ['pink', 'green', 'Beige-pink corn', 'Beige-yellow corn', 'yellow', 'Beige-green corn']}}, {{'name': 'Specification', 'values': ['42-45cm', '52-55cm', '32-35cm']}}]
The following are several variants of the product:
Variant name: Flower Plush Throw Pillow Soft Plant Cartoon Chair Cushion Living Bedroom Home Decorative Pillows Sofa Cushions Birthday Gifts - pink / 42-45cm, Variant ID: gid://shopify/ProductVariant/45802066149606, Price: 18.99, Inventory Quantity: 0
Variant name: Flower Plush Throw Pillow Soft Plant Cartoon Chair Cushion Living Bedroom Home Decorative Pillows Sofa Cushions Birthday Gifts - pink / 52-55cm, Variant ID: gid://shopify/ProductVariant/45802066182374, Price: 24.99, Inventory Quantity: 0
Variant name: Flower Plush Throw Pillow Soft Plant Cartoon Chair Cushion Living Bedroom Home Decorative Pillows Sofa Cushions Birthday Gifts - green / 32-35cm, Variant ID: gid://shopify/ProductVariant/45802066215142, Price: 19.99, Inventory Quantity: 0

product_experience_level: new to this product
customer_type: recent customer
persona: explorative
current_webpage: product page
modality: visual
communication_type: digital_preference
discovery_type: search engine results
buying_behavior: value-conscious explorer
budget: low-budget
purchase_history: Explorative purchase history with a focus on home goods and occasional interest in apparel.
Location: New York City, NY, USA

Thought:
The user has placed two orders: gid://shopify/Order/6294747119846 and gid://shopify/Order/6294747807974, however gid://shopify/Order/6294747807974 has been cancelled, so the user want to track the other order.

Attribute:
Order id: gid://shopify/Order/6294747119846

########################################################
Goal: {goal}
Full attributes: 
{user_profile}

"""

        system_instruction = FIND_MATCHED_ATTRIBUTE_PROMPT.format(
            goal=goal, user_profile=user_profile_str
        )
        print(system_instruction)
        response = chatgpt_chatbot(
            [{"role": "user", "content": system_instruction}], client=client
        )
        thought = response.split("Thought:")[1].split("Attribute:")[0].strip()
        print("thought: ", thought)
        attribute = response.split("Attribute:")[1].strip()
        print("attribute: ", attribute)
    else:
        raise ValueError("Invalid strategy")
    return attribute


def pick_attributes(
    user_profile: AttributeDict,
    attributes: dict[str, list[str]],
    goals: GoalList,
    strategy: StrategyType = "react",
    client: OpenAI | anthropic.Anthropic | None = None,
) -> tuple[AttributeDict, LabelDict]:
    """Pick attributes based on user profile and goals using specified strategy.

    This function selects appropriate attributes for a user profile based on the
    available goals and the specified selection strategy.

    Args:
        user_profile: Dictionary containing current user profile attributes.
        attributes: Dictionary mapping attribute categories to lists of possible values.
        goals: List of available goals to choose from.
        strategy: Strategy for attribute selection. Options:
            - "react": Use reactive strategy with LLM reasoning
            - "random": Use random selection
        client: LLM client for attribute selection (required for reactive strategy).

    Returns:
        Tuple containing:
        - Selected attributes dictionary
        - Matched attribute to goal mapping for labeling
    """
    if strategy == "react":
        return pick_attributes_react(user_profile, attributes, goals, client)
    else:
        return pick_attributes_random(user_profile, attributes, goals, client)


def _select_random_attributes(
    attributes: dict[str, list[str]], goals: GoalList
) -> tuple[AttributeDict, str]:
    """Select random attributes and goal from available options.

    Args:
        attributes: Dictionary mapping attribute categories to lists of possible values.
        goals: List of available goals to choose from.

    Returns:
        Tuple containing:
        - Selected attributes dictionary with randomly chosen values
        - Randomly selected goal string
    """
    selected_attributes: AttributeDict = {}
    goal = random.choice(goals)
    selected_attributes["goal"] = goal

    for category, values in attributes.items():
        if category == "goal":
            continue
        # Handle empty values by providing a default value
        if not values:
            # For empty values, use a generic default based on the category
            if "budget" in category.lower():
                selected_attributes[category] = "medium"
            elif "location" in category.lower():
                selected_attributes[category] = "United States"
            elif "history" in category.lower():
                selected_attributes[category] = "none"
            elif "job" in category.lower():
                selected_attributes[category] = "professional"
            elif "business" in category.lower():
                selected_attributes[category] = "small business"
            elif "size" in category.lower():
                selected_attributes[category] = "10-50 employees"
            else:
                selected_attributes[category] = "default"
        else:
            selected_attributes[category] = random.choice(values)

    return selected_attributes, goal


def pick_attributes_react(
    user_profile: AttributeDict,
    attributes: dict[str, list[str]],
    goals: GoalList,
    client: OpenAI | anthropic.Anthropic,
) -> tuple[AttributeDict, LabelDict]:
    """Pick attributes using a reactive strategy with LLM reasoning.

    This function uses random selection for initial attributes but then uses
    LLM reasoning to find the most relevant attribute for the selected goal.

    Args:
        user_profile: Dictionary containing current user profile attributes.
        attributes: Dictionary mapping attribute categories to lists of possible values.
        goals: List of available goals to choose from.
        client: LLM client for attribute matching.

    Returns:
        Tuple containing:
        - Selected attributes dictionary
        - Matched attribute to goal mapping for labeling
    """
    selected_attributes, _ = _select_random_attributes(attributes, goals)
    matched_attribute_to_goal = find_matched_attribute(
        selected_attributes["goal"], selected_attributes, client=client
    )
    return selected_attributes, matched_attribute_to_goal


def pick_attributes_random(
    user_profile: AttributeDict,
    attributes: dict[str, list[str]],
    goals: GoalList,
    client: OpenAI | anthropic.Anthropic | None = None,
) -> tuple[AttributeDict, LabelDict]:
    """Pick attributes randomly from available options.

    This function randomly selects attributes and goals without using LLM reasoning.

    Args:
        user_profile: Dictionary containing current user profile attributes.
        attributes: Dictionary mapping attribute categories to lists of possible values.
        goals: List of available goals to choose from.
        client: LLM client (not used in random strategy).

    Returns:
        Tuple containing:
        - Randomly selected attributes dictionary
        - Matched attribute to goal mapping for labeling
    """
    selected_attributes, _ = _select_random_attributes(attributes, goals)
    matched_attribute_to_goal = find_matched_attribute(
        selected_attributes["goal"], selected_attributes, client=client
    )
    return selected_attributes, matched_attribute_to_goal


def adapt_goal(goal: str, config: ConfigDict, doc: str, user_profile: str) -> str:
    """Adapt a goal to be more specific to the company and user profile.

    This function uses LLM to rewrite a generic goal into one that is more
    specific to the company context and aligned with the user's profile.

    Args:
        goal: The original goal to be adapted.
        config: Configuration dictionary containing intro/company_summary and client.
        doc: Company document content for context.
        user_profile: Formatted user profile string.

    Returns:
        Adapted goal string that is more specific to the company context.
    """
    # Use 'intro' if available, otherwise fall back to 'company_summary'
    company_summary = config.get("intro", config.get("company_summary", ""))
    prompt = ADAPT_GOAL.format(
        goal=goal,
        company_summary=company_summary,
        doc=doc,
        user_profile=user_profile,
    )
    adapted_goal = chatgpt_chatbot(
        [{"role": "user", "content": prompt}], config["client"]
    )
    return adapted_goal


def _fetch_api_data(api_url: str, key: str) -> list[Any]:
    """Fetch data from API with error handling and timeout.

    Args:
        api_url: URL of the API endpoint to fetch data from.
        key: Key for logging purposes when errors occur.

    Returns:
        List of data from the API response, or empty list if request fails.

    Note:
        Uses a 10-second timeout and handles both network and JSON parsing errors.
    """
    try:
        response = requests.get(api_url, timeout=10).json()
        return response
    except (requests.RequestException, ValueError) as e:
        log_context.error(f"Failed to fetch data for {key}: {e}")
        # Re-raise ValueError to maintain expected behavior for tests
        if isinstance(e, ValueError):
            raise
        return []


def get_custom_profiles(
    config: ConfigDict,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Get custom profiles from the database with API integration and binding support.

    This function fetches user profiles and system attributes from external APIs
    and handles binding relationships between different attribute categories.

    Args:
        config: Configuration dictionary containing user_attributes with API endpoints
            and binding configurations.

    Returns:
        Tuple containing:
        - Dictionary of user profiles fetched from APIs
        - Dictionary of system attributes fetched from APIs

    Note:
        Supports binding relationships where selecting one attribute automatically
        selects related attributes from the same API response.
    """
    user_profiles: dict[str, Any] = {}
    system_attributes: dict[str, Any] = {}

    if (
        "system_attributes" in config["user_attributes"]
        and "user_profiles" in config["user_attributes"]
    ):
        bindings: dict[str, Any] = {}

        for key, value in config["user_attributes"]["system_attributes"].items():
            full_key = f"system_attributes.{key}"
            if isinstance(value, dict):
                api_url = value.get("api")
                response = _fetch_api_data(api_url, key)
                system_attributes[key] = response
                if "bind_to" in value:
                    bindings[full_key] = response
                    bindings[value["bind_to"]] = response
            else:
                system_attributes[key] = value

        for key, value in config["user_attributes"]["user_profiles"].items():
            full_key = f"user_profiles.{key}"
            if isinstance(value, dict):
                if "bind_to" in value and value["bind_to"] in bindings:
                    user_profiles[key] = bindings[value["bind_to"]]
                else:
                    api_url = value.get("api")
                    response = _fetch_api_data(api_url, key)
                    user_profiles[key] = response
            else:
                user_profiles[key] = value

    return user_profiles, system_attributes


def _build_tool_list(config: ConfigDict) -> tuple[list[dict[str, Any]], Any]:
    """Build list of available tools for selection with environment setup.

    This function creates a list of available tools from the configuration and
    sets up the environment for tool execution. It also adds a "no tool" option
    for cases where no appropriate tool exists.

    Args:
        config: Configuration dictionary containing tools and workers lists.

    Returns:
        Tuple containing:
        - List of tool dictionaries with tool_id, description, input, and output
        - Environment object for tool execution
    """
    env = Environment(
        tools=config["tools"],
        workers=config["workers"],
        agents=config.get("agents", []),
    )
    tool_list: list[dict[str, Any]] = []

    for tool in config["tools"]:
        tool_id = tool["id"]
        tool_obj: Tool = env.tools[tool_id]["execute"]()
        slots = tool_obj.slots
        tool_description = tool_obj.description
        tool_input = [s.model_dump() for s in slots]
        tool_output = tool_obj.output
        tool_list.append(
            {
                "tool_id": tool_id,
                "tool_description": tool_description,
                "tool_input": tool_input,
                "tool_output": tool_output,
            }
        )

    tool_list.append(
        {
            "tool_id": "0",
            "tool_description": "There are no tools appropriate for the goal.",
            "tool_input": [],
            "tool_output": "There are no tools appropriate for the goal.",
        }
    )

    return tool_list, env


def get_label(
    attribute: AttributeDict, config: ConfigDict
) -> tuple[list[dict[str, Any]], bool]:
    """Get the appropriate tool and slot values for achieving the user's goal.

    This function uses LLM reasoning to select the most appropriate tool for
    achieving a given goal and fills in the required slot values based on the
    user's attributes.

    Args:
        attribute: Dictionary containing user attributes including the goal.
        config: Configuration dictionary containing tools, workers, and client.

    Returns:
        Tuple containing:
        - List of tool dictionaries with tool_id, tool_name, and filled slots
        - Boolean indicating success (always True in current implementation)

    Note:
        Includes retry logic (up to 3 attempts) for handling tool selection errors.
        Falls back to "no tool" option if all attempts fail.
    """
    GET_TOOL_PROMPT = """Given the list of tools that an AI assistant can use, and the user's goal, return the tool that is most likely to be used to achieve the goal. Only return the tool id.
    Tools: {tools}
    User's goal: {goal}
    Tool_id:
    """

    tool_list, env = _build_tool_list(config)

    label: list[dict[str, Any]] = [
        {
            "tool_id": "0",
            "tool_name": "No tool",
            "slots": {},
        }
    ]

    attempt = 0
    while attempt < 3:
        try:
            response = chatgpt_chatbot(
                [
                    {
                        "role": "system",
                        "content": GET_TOOL_PROMPT.format(
                            tools="\n".join(
                                f"tool_id: {tool['tool_id']}\ntool_description: {tool['tool_description']}\ntool_input: {tool['tool_input']}\ntool_output: {tool['tool_output']}"
                                for tool in tool_list
                            ),
                            goal=attribute["goal"],
                        ),
                    }
                ],
                config["client"],
            )
            pred_tool_id = response
            if pred_tool_id == "0":
                break

            selected_tool = env.tools[pred_tool_id]["execute"]()
            slots = selected_tool.slots
            pred_slots = SlotFiller(url="").execute(
                slots, str(attribute), type="user_simulator"
            )
            pred_slots_dict = {slot.name: slot.value for slot in pred_slots}
            label = [
                {
                    "tool_id": pred_tool_id,
                    "tool_name": selected_tool.name,
                    "slots": pred_slots_dict,
                }
            ]
            break
        except (KeyError, ValueError, AttributeError) as e:
            log_context.warning(f"Error in tool selection attempt {attempt + 1}: {e}")
            attempt += 1
        except Exception as e:
            log_context.error(
                f"Unexpected error in tool selection attempt {attempt + 1}: {e}"
            )
            attempt += 1

    return label, True


def filter_attributes(config: ConfigDict) -> dict[str, dict[str, Any]]:
    """Filter attributes from the configuration, excluding 'goal' and 'system_attributes'.

    Args:
        config: Configuration dictionary containing user_attributes.

    Returns:
        Dictionary of predefined attributes excluding goal and system attributes.
    """
    predefined_attributes: dict[str, dict[str, Any]] = {}
    for key, value in config["user_attributes"].items():
        if key != "goal" and key != "system_attributes":
            predefined_attributes[key] = value
    return predefined_attributes


def select_system_attributes(
    config: ConfigDict, synthetic_data_params: dict[str, Any]
) -> list[SystemAttributeDict]:
    """Select system attributes for each conversation from the configuration.

    Args:
        config: Configuration dictionary containing user_attributes.system_attributes.
        synthetic_data_params: Parameters containing num_convos for the number of
            conversations to generate system attributes for.

    Returns:
        List of system attribute dictionaries, one for each conversation.
    """
    system_attributes: list[SystemAttributeDict] = []
    for _ in range(synthetic_data_params["num_convos"]):
        system_attribute: SystemAttributeDict = {}
        for key, value in config["user_attributes"]["system_attributes"].items():
            system_attribute[key] = random.choice(value["values"])
        system_attributes.append(system_attribute)
    return system_attributes


def augment_attributes(
    predefined_attributes: dict[str, dict[str, Any]],
    config: ConfigDict,
    documents: list[DocumentType],
) -> dict[str, list[str]]:
    """Augment attributes with additional values using LLM if needed.

    This function can extend predefined attribute values with additional options
    generated by LLM based on company information and documents.

    Args:
        predefined_attributes: Dictionary of predefined attributes with their values.
        config: Configuration dictionary containing intro/company_summary and client.
        documents: List of company documents for context.

    Returns:
        Dictionary mapping attribute categories to lists of values (original + augmented).

    Note:
        Only augments attributes that have the "augment" flag set to True in their
        configuration. Uses company documents for context when available.
    """
    augmented_attributes: dict[str, list[str]] = {}
    # Use 'intro' if available, otherwise fall back to 'company_summary'
    company_summary = config.get("intro", config.get("company_summary", ""))

    for category, category_data in predefined_attributes.items():
        # Handle nested structure where category contains subcategories
        if isinstance(category_data, dict) and "values" not in category_data:
            # This is a nested category (like "generic", "b2b", "b2c")
            # Flatten the nested structure
            for subcategory, subcategory_data in category_data.items():
                if isinstance(subcategory_data, dict) and "values" in subcategory_data:
                    augmented_attributes[subcategory] = subcategory_data["values"]
                    if "augment" in subcategory_data and subcategory_data["augment"]:
                        doc = ""
                        if documents and len(documents) > 0:
                            doc = (
                                "Here is a page from the company website: "
                                + random.choice(documents)["content"]
                            )
                            prompt = ADD_ATTRIBUTES.format(
                                user_profile="; ".join(
                                    f"{key}: {value}"
                                    for key, value in predefined_attributes.items()
                                ),
                                category=subcategory,
                                company_summary=company_summary,
                                company_doc=doc,
                            )
                        else:
                            prompt = ADD_ATTRIBUTES_WO_DOC.format(
                                user_profile="; ".join(
                                    f"{key}: {value}"
                                    for key, value in predefined_attributes.items()
                                ),
                                category=subcategory,
                                company_summary=company_summary,
                            )
                        response = chatgpt_chatbot(
                            [{"role": "user", "content": prompt}], config["client"]
                        )
                        new_values = [value.strip() for value in response.split(",")]
                        augmented_attributes[subcategory].extend(new_values)
        else:
            # This is a flat category with direct "values" key
            augmented_attributes[category] = category_data["values"]
            if "augment" in category_data and category_data["augment"]:
                doc = ""
                if documents and len(documents) > 0:
                    doc = (
                        "Here is a page from the company website: "
                        + random.choice(documents)["content"]
                    )
                    prompt = ADD_ATTRIBUTES.format(
                        user_profile="; ".join(
                            f"{key}: {value}"
                            for key, value in predefined_attributes.items()
                        ),
                        category=category,
                        company_summary=company_summary,
                        company_doc=doc,
                    )
                else:
                    prompt = ADD_ATTRIBUTES_WO_DOC.format(
                        user_profile="; ".join(
                            f"{key}: {value}"
                            for key, value in predefined_attributes.items()
                        ),
                        category=category,
                        company_summary=company_summary,
                    )
                response = chatgpt_chatbot(
                    [{"role": "user", "content": prompt}], config["client"]
                )
                new_values = [value.strip() for value in response.split(",")]
                augmented_attributes[category].extend(new_values)
    return augmented_attributes


def convert_attributes_to_profiles(
    attributes_list: list[AttributeDict],
    system_attributes_list: list[SystemAttributeDict],
    config: ConfigDict,
) -> tuple[ProfileList, GoalList, list[SystemAttributeDict]]:
    """Convert attributes to profile descriptions and extract goals.

    Args:
        attributes_list: List of attribute dictionaries for each conversation.
        system_attributes_list: List of system attribute dictionaries for each conversation.
        config: Configuration dictionary containing company_summary and client.

    Returns:
        Tuple containing:
        - List of profile description strings
        - List of goal strings
        - List of system input dictionaries
    """
    profiles: ProfileList = []
    goals: GoalList = []
    system_inputs: list[SystemAttributeDict] = []

    for attributes, system_attributes in zip(
        attributes_list, system_attributes_list, strict=False
    ):
        profile = convert_attributes_to_profile(attributes, config)
        profiles.append(profile)
        goals.append(attributes["goal"])
        system_inputs.append(system_attributes)

    return profiles, goals, system_inputs


def convert_attributes_to_profile(attributes: AttributeDict, config: ConfigDict) -> str:
    """Convert a single attribute dictionary to a profile description.

    Args:
        attributes: Dictionary containing user attributes.
        config: Configuration dictionary containing intro/company_summary and client.

    Returns:
        Formatted profile description string generated by LLM.
    """
    user_attr = "; ".join(f"{key}: {value}" for key, value in attributes.items())
    # Use 'intro' if available, otherwise fall back to 'company_summary'
    company_summary = config.get("intro", config.get("company_summary", ""))
    prompt = ATTR_TO_PROFILE.format(
        company_summary=company_summary, user_attr=user_attr
    )
    profile = chatgpt_chatbot([{"role": "user", "content": prompt}], config["client"])
    return profile


def build_user_profiles(test_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build user profiles from test data.

    Args:
        test_data: List of test data dictionaries.

    Returns:
        List of user profile dictionaries (currently returns empty list).
    """
    return []


def attributes_to_text(attribute_list: list[AttributeDict]) -> list[str]:
    """Convert a list of attribute dictionaries to a list of formatted text strings.

    Args:
        attribute_list: List of attribute dictionaries to convert.

    Returns:
        List of formatted text strings, one for each attribute dictionary.
        Each string contains key-value pairs separated by newlines.
    """
    text_attributes: list[str] = []
    for item in attribute_list:
        text_attribute = ""
        for key, value in item.items():
            text_attribute += f"{key}: {value}\n"
        text_attributes.append(text_attribute[:-1])  # Remove trailing newline
    return text_attributes
