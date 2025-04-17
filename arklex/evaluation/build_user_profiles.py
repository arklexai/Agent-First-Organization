import json
import random
import requests
import copy
from arklex.evaluation.get_documents import load_docs
from arklex.evaluation.chatgpt_utils import (chatgpt_chatbot, query_chatbot, filter_convo, adjust_goal,
                                               flip_hist, generate_goals, format_chat_history_str, flip_hist_content_only)
from arklex.env.env import Env
from arklex.orchestrator.NLU.nlu import SlotFilling
from arklex.env.tools.tools import Tool

ATTR_TO_PROFILE = "Convert the following list user attributes in to a text description of a customer profile for the following company:\n{company_summary}\nThe user attributes are here:\n{user_attr}"
ADAPT_GOAL = "Assume you are planning to speak to a chatbot with the following goal in mind:\n{goal}\nUsing the company information below, re-write this goal into one that is more specific to the company. The new goal should mention specific products (if relevant) or other details about the company. Here is a summary of the company:\n{company_summary}\nHere is a page from the company website:\n{company_doc}"
ADD_ATTRIBUTES = "Your job is to add attributes to a customer profile. Here is an example of an existing profile with the categories on the left and the attributes on the right:\n{user_profile}\nSuggest three attributes for the following category:\n{category}\nThese attributes should be specific values that are relevant to the category and apply to potential customers of the company. You should return a comma separated list of attributes without any descriptions of the attributes. Generated the attributes based on a summary of the company and the company webpage and what kind of customers the compnay is likely targeting. Here is the summary fo the company:\n{company_summary}\nHere is the webpage:\n{company_doc}"


def build_profile(synthetic_data_params, config) -> tuple[list[str], list[str], list[dict], list[dict], list[dict]]:
    labels_list = []
    if not config['custom_profile']: # Use predefined profiles (from user_attributes.json)
        documents = load_docs(config['documents_dir'], config, synthetic_data_params['num_goals'] * 2)
        attributes_list = filter_attributes(config)
        attributes_list = augment_attributes(attributes_list, config, documents)
        attributes_list = select_attributes(attributes_list, synthetic_data_params)
        if config['system_inputs']:
            system_attributes_list = select_system_attributes(config, synthetic_data_params)
        else:
            system_attributes_list = [{}] * synthetic_data_params['num_convos']
        attributes_list_with_goals = adapt_goals(attributes_list, config, documents)
        profiles, goals, system_inputs = convert_attributes_to_profiles(attributes_list_with_goals, system_attributes_list, config)
        return profiles, goals, attributes_list, system_inputs, labels_list

    else: # Use custom profiles (from database)
        user_profiles, system_attributes = get_custom_profiles(config)
        predefined_attributes = filter_attributes(config)
        system_attributes_list = []
        attributes_list = []
        labels_list = []
        for i in range(synthetic_data_params['num_convos']):
            system_attribute = {}
            user_profile = {}
            binding_index = {}
            for key, value in system_attributes.items():
                full_key = f"system_attributes.{key}"
                if "bind_to" in value:
                    random_index = random.choice(range(len(system_attributes[key])))
                    system_attribute[key] = system_attributes[key][random_index]
                    binding_index[value["bind_to"]] = random_index
                else:
                    random_index = random.choice(range(len(system_attributes[key])))
                    system_attribute[key] = system_attributes[key][random_index]
        
            for key, value in user_profiles.items():
                full_key = f"user_profiles.{key}"
                if "bind_to" in value and full_key in binding_index:
                    user_profile[key] = user_profiles[key][binding_index[full_key]]
                if "bind_to" not in value:
                    random_index = random.choice(range(len(user_profiles[key])))
                    user_profile[key] = user_profiles[key][random_index]
            # based on the user's profile, select the attribute
            attribute = pick_attribute(user_profile, predefined_attributes, config['client'])
            # get the proposed tool from the goal and the corresponding input as label
            label, valid = get_label(attribute, config)
            
            labels_list.append(label)
            attributes_list.append(attribute)
            system_attributes_list.append(system_attribute)
        
        profiles, goals, system_inputs = convert_attributes_to_profiles(attributes_list, system_attributes_list, config)
        
    return profiles, goals, attributes_list, system_inputs, labels_list


def pick_attribute(user_profile, predefined_attributes, client):
    """
    Pick the attribute from the predefined attributes based on the user's profile to avoid the attribute confliction
    """
    attributes = {}
    for key, value in predefined_attributes.items():
        PICK_ATTRIBUTE_PROMPT = """Given the user's profile, please pick or modify the attribute value from the choice list of the given attribute category {category}. If all the given choices are not align with the user's profile, then generate a new attribute value that is most likely to be used by the user. Please directly return the attribute value without any description.
        User's profile: {user_profile}
        {category}: {choices}
        Attribute value:
        """
        response = chatgpt_chatbot([{'role': 'system', 'content': PICK_ATTRIBUTE_PROMPT.format(user_profile=user_profile, category=key, choices="\n".join(value))}], client)
        attributes[key] = response
        
    return attributes


def get_custom_profiles(config) -> tuple[dict, dict]:

    if "system_attributes" in config["user_attributes"] and "user_profiles" in config["user_attributes"]:
        # First, get system attributes with bindings
        system_attributes = {}
        bindings = {}  # Track bindings between fields
        
        # Process system_attributes and their bindings
        for key, value in config["user_attributes"]["system_attributes"].items():
            full_key = f"system_attributes.{key}"
            if isinstance(value, dict):
                api_url = value.get("api")
                response = requests.get(api_url).json()
                system_attributes[key] = response
                # Track bindings if they exist
                if "bind_to" in value:
                    bindings[full_key] = response
                    bindings[value["bind_to"]] = response
            else:
                system_attributes[key] = value
    
        user_profiles = {}
        # Process user_profiles and their bindings
        for key, value in config["user_attributes"]["user_profiles"].items():
            if isinstance(value, dict):
                if "bind_to" in value and value["bind_to"] in bindings:
                    user_profiles[key] = bindings[value["bind_to"]]
                else:
                    api_url = value.get("api")
                    response = requests.get(api_url).json()
                    user_profiles[key] = response
            else:
                user_profiles[key] = value
    
    return user_profiles, system_attributes


def get_label(attribute, config):
    """
    Get the appropriate tool used by the Agent to achieve the user's goal
    """
    valid = True   #dummy variable
    GET_TOOL_PROMPT = """Given the list of tools that an AI assistant can use, and the user's goal, return the tool that is most likely to be used to achieve the goal. Only return the tool id.
    Tools: {tools}
    User's goal: {goal}
    Tool_id:
    """
    env = Env(
        tools = config["tools"],
        workers = config["workers"]
    )
    tool_list = []
    for tool in config["tools"]:
        tool_id = tool["id"]
        tool: Tool = env.tools[tool_id]["execute"]()
        slots = tool.slots
        tool_description = tool.description
        tool_input =[s.model_dump() for s in slots]
        tool_output = tool.output
        tool_list.append({
            "tool_id": tool_id,
            "tool_description": tool_description,
            "tool_input": tool_input,
            "tool_output": tool_output
        })
    tool_list.append({
        "tool_id": "0",
        "tool_description": "There are no tools appropriate for the goal.",
        "tool_input": [],
        "tool_output": "There are no tools appropriate for the goal."
    })

    label = [
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
                [{'role': 'system', 'content': GET_TOOL_PROMPT.format(tools="\n".join(f"tool_id: {tool['tool_id']}\ntool_description: {tool['tool_description']}\ntool_input: {tool['tool_input']}\ntool_output: {tool['tool_output']}" for tool in tool_list), goal=attribute["goal"])}],
                config['client']
            )
            pred_tool_id = response
            if pred_tool_id == "0":
                break
            selected_tool = env.tools[pred_tool_id]["execute"]()
            slots = selected_tool.slots
            pred_slots = SlotFilling(url="").execute(slots, str(attribute), type="user_simulator")
            pred_slots_dict = {slot.name: slot.value for slot in pred_slots}
            label = [
                {
                    "tool_id": pred_tool_id,
                    "tool_name": selected_tool.name,
                        "slots": pred_slots_dict,
                }
            ]
            break
        except Exception as e:
            attempt += 1


    return label, valid

def filter_attributes(config) -> dict:
    filtered_attributes = {}
    for key in config['user_attributes'].keys():
        if key == 'generic' or key == config['synthetic_data_params']['customer_type']:
            for subkey in config['user_attributes'][key].keys():
                filtered_attributes[subkey] = config['user_attributes'][key][subkey]
    return filtered_attributes

def select_system_attributes(config, synthetic_data_params) -> list[dict[str, dict]]:
    system_attributes = []
    for subkey, subvalue in config["user_attributes"]["system_attributes"].items():
        if isinstance(subvalue, dict):
            api_url = subvalue.get("api")
            response = requests.get(api_url).json()
            config["user_attributes"]["system_attributes"][subkey] = response

    for i in range(synthetic_data_params['num_convos']):
        system_attribute = {}
        for subkey, subvalue in config["user_attributes"]["system_attributes"].items():
            if isinstance(subvalue, list) and isinstance(subvalue[0], dict):
                system_attribute[subkey] = random.choice(subvalue)
            else:
                raise ValueError("System attributes should be a list of dictionaries")
        system_attributes.append(copy.deepcopy(system_attribute))
    return system_attributes


def select_attributes(user_attributes, synthetic_data_params):
    user_list = []
    for i in range(synthetic_data_params['num_convos']):
        attributes = {}
        for key, value in user_attributes.items():
            attributes[key] = random.choice(value)
        user_list.append(attributes.copy())
    return user_list

def augment_attributes(attributes_list, config, documents):
    # add attribute values using docs 
    new_attrs = generate_attributes(attributes_list, config, documents)
    return new_attrs

def adapt_goals(attributes_list, config, documents):
    attributes_list_with_goals = []
    for item in attributes_list:
        new_goal = adapt_goal(item['goal'], config, documents)
        new_item = {}
        for key in item.keys():
            if key == 'goal':
                new_item['goal'] = new_goal
            else:
                new_item[key] = item[key]
        attributes_list_with_goals.append(new_item)
    return attributes_list_with_goals

def adapt_goal(goal, config, documents):
    new_goal = chatgpt_chatbot([{'role': 'user', 'content': ADAPT_GOAL.format(goal=goal, company_summary=config['intro'], company_doc=random.choice(documents))}], config['client'])
    return new_goal

def generate_attributes(attributes, config, documents):
    text_attribute = ''
    for key, value in attributes.items():
        if len(value['values']) == 0:
            continue
        text_attribute += f"{key}: {value['values']}\n"

    new_attrs = {}
    for category in attributes.keys():
        if not attributes[category]['generate_values']:
            new_attrs[category] = attributes[category]['values']
        else:
            attrs = chatgpt_chatbot([{'role': 'user', 'content': ADD_ATTRIBUTES.format(user_profile=text_attribute, category=category, company_summary=config['intro'], company_doc=random.choice(documents))}], config['client'])
            new_attrs[category] = attrs.split(', ')
    return new_attrs

def attributes_to_text(attribute_list):
    text_attributes = []
    for item in attribute_list:
        text_attribute = ''
        for key, value in item.items():
            text_attribute += f"{key}: {value}\n"
        text_attributes.append(text_attribute[:-1])
    return text_attributes

def convert_attributes_to_profiles(attributes_list, system_attributes, config, mode="full"):
    profile_list = []
    system_inputs = []
    for sys_attr, attr_list in zip(system_attributes, attributes_list):
        system_input = {}
        for key, value in sys_attr.items():
            if mode == "full":
                attr_list[key] = value["attribute"]
            system_input[key] = value["input"]
        system_inputs.append(system_input)

    text_attributes = attributes_to_text(attributes_list)
    for i, attribute in enumerate(text_attributes):
        profile = chatgpt_chatbot([{'role': 'user', 'content': ATTR_TO_PROFILE.format(company_summary=config['intro'], user_attr=attribute)}], config['client'])
        profile_list.append({"profile": profile, "goal": attributes_list[i]["goal"]})
    
    profiles = [item['profile'] for item in profile_list]
    goals = [item['goal'] for item in profile_list]
    return profiles, goals, system_inputs


def build_labelled_profile(synthetic_data_params, config):
    env = Env(
        tools = config["tools"],
        workers = config["workers"]
    )
    
    user_profiles, system_attributes = get_custom_profiles(config, synthetic_data_params)
    labels_list = []
    attributes_list_with_goals = []
    system_attributes_list = []
    for i in range(synthetic_data_params['num_convos']):
        system_attribute = {}
        user_profile = {}
        binding_index = {}
        for key, value in system_attributes.items():
            full_key = f"system_attributes.{key}"
            if "bind_to" in value:
                random_index = random.choice(range(len(system_attributes[key])))
                system_attribute[key] = system_attributes[key][random_index]
                binding_index[value["bind_to"]] = random_index
            else:
                random_index = random.choice(range(len(system_attributes[key])))
                system_attribute[key] = system_attributes[key][random_index]
    
        for key, value in user_profiles.items():
            full_key = f"user_profiles.{key}"
            if "bind_to" in value and full_key in binding_index:
                user_profile[key] = user_profiles[key][binding_index[full_key]]
            if "bind_to" not in value:
                random_index = random.choice(range(len(user_profiles[key])))
                user_profile[key] = user_profiles[key][random_index]
        # option1: 
        # randomly choose any one of tools from the whole tool list for each conversation
        # select the direct slot value from the user's profile
        while True:
            # slot filling
            tool_id = random.choice(config["tools"])["id"]
            tool: Tool = env.tools[tool_id]["execute"]()
            slots = tool.slots
            pred_slots = SlotFilling(url="").execute(slots, str(user_profile), metadata={}, type="user_simulator")
            # if all slots are filled, means that the user's profile has enough information to complete the goal
            # if not, try again
            if all(slot.value for slot in pred_slots):
                break
        pred_slots_dict = {slot.name: slot.value for slot in pred_slots}
        label = [
            {
                "tool_id": tool_id,
                "tool_name": tool.name,
                    "slots": pred_slots_dict,
            }
        ]
        labels_list.append(label)

        tool_description = tool.description
        tool_input =[s.model_dump() for s in pred_slots]
        tool_output = tool.output
        goal_generation_prompt = f"""Given a tool that an AI assistant can use, imagine what kind of user query or request would naturally require this tool to fulfill it.

Tool Description: {tool_description}
Tool Input: {tool_input}
Tool Output: {tool_output}

Think about:
1. What problem or need would a user have that this tool could solve?
2. What would the user naturally ask for, without knowing this tool exists?
3. What real-world goal would lead to using this tool?

Generate a natural high-level user goal that would require this tool to fulfill it. No example or detailed content is needed.
Write it as a first-person statement starting with "I want" or "I need". A user can only do the action on themselves. Cannot manipulate the tool to do the action on other people.
The goal should within 15 words.


User's goal:"""
        goal = chatgpt_chatbot(
            [{'role': 'system', 'content': goal_generation_prompt}],
            config['client']
        )
        attributes_list_with_goals.append({"goal": goal, **pred_slots_dict})
        print("+++++++++++++++++++++++++++")
        print("tool_description: ", tool_description)
        print("goal: ", goal)
        print("+++++++++++++++++++++++++++")

        system_attributes_list.append(system_attribute)
    
    
    profiles, goals, system_inputs = convert_attributes_to_profiles(attributes_list_with_goals, system_attributes_list, config, mode="separate")
    
    
    return profiles, goals, attributes_list_with_goals, system_inputs, labels_list
    