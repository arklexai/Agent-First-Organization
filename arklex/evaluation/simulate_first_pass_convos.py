"""First-pass conversation simulation for evaluation in the Arklex framework.

This module provides functionality for simulating and evaluating conversations between users
and the chatbot system. It includes utilities for profile matching, conversation generation,
goal completion checking, and handling different user attributes and behaviors. The module
supports both synthetic and example-based conversation generation, with parallel processing
capabilities for efficient evaluation.
"""

import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from arklex.evaluation.build_user_profiles import ATTR_TO_PROFILE, build_profile
from arklex.evaluation.chatgpt_utils import (
    chatgpt_chatbot,
    filter_convo,
    flip_hist,
    flip_hist_content_only,
    format_chat_history_str,
    query_chatbot,
)

# USER_DATA_KEYS = ['goal', 'product_experience_level', 'deal_stage', 'customer_type', 'decision_making_authority', 'persona', 'discovery_type', 'buying_behavior']
USER_DATA_KEYS: list[str] = [
    "goal",
    "product_experience_level",
    "customer_type",
    "persona",
    "discovery_type",
    "buying_behavior",
]


def get_relevant_vals(attr: dict[str, Any]) -> list[str]:
    vals: list[str] = []
    for key in USER_DATA_KEYS:
        vals.append(attr[key])
    return vals


def count_matches(l1: list[str], l2: list[str]) -> int:
    num_matches: int = 0
    for i in range(len(l1)):
        if l1[i] == l2[i]:
            num_matches += 1
    return num_matches


def join_messages(messages: list[dict[str, Any]]) -> str:
    message_str: str = ""
    for message in messages:
        if message["role"] == "bot_follow_up":
            continue
        message_str += f"{message['role']}: {message['content']}\n"
    return message_str[:-1]


def create_convo_profile(
    best_match: list[str], attr_vals: list[str], summary: str, client: object
) -> str:
    dict_profile: dict[str, str] = {}
    for i in range(len(USER_DATA_KEYS)):
        if best_match[i] == "other":
            continue
        dict_profile[USER_DATA_KEYS[i]] = best_match[i]

    text_profile: str = ""
    for key, value in dict_profile.items():
        text_profile += f"{key}: {value}\n"
    profile: str = chatgpt_chatbot(
        [
            {
                "role": "user",
                "content": ATTR_TO_PROFILE.format(
                    company_summary=summary, user_attr=text_profile[:-1]
                ),
            }
        ],
        client,
    )
    return profile


def retrieve_convo(
    attr_vals: list[str],
    all_profiles: list[str],
    user_convos: dict[str, list[dict[str, Any]]],
    summary: str,
    client: object,
) -> tuple[str, str]:
    split_profiles: list[list[str]] = [p.split(",") for p in all_profiles]
    best_match: list[str] | None = None
    max_matches: int = 0
    for profile in split_profiles:
        num_matches: int = count_matches(attr_vals, profile)
        if num_matches >= max_matches:
            best_match = profile
            max_matches = num_matches
            num_matches = 0

    convo: dict[str, Any] = random.choice(user_convos[",".join(best_match)])
    convo_messages: str = join_messages(convo["message"])
    convo_profile: str = create_convo_profile(best_match, attr_vals, summary, client)
    return convo_messages, convo_profile


def get_example_convo(
    attr: dict[str, Any],
    synthetic_data_params: dict[str, Any],
    summary: str,
    client: object,
) -> tuple[str, str]:
    with open(synthetic_data_params["data_file"]) as f:
        user_convos: dict[str, list[dict[str, Any]]] = json.load(f)

    all_profiles: list[str] = list(user_convos.keys())
    attr_vals: list[str] = get_relevant_vals(attr)
    convo, matching_profile = retrieve_convo(
        attr_vals, all_profiles, user_convos, summary, client
    )
    return convo, matching_profile


def retrieve_prompts(
    profile: str,
    goal: str,
    attr: dict[str, Any],
    summary: str,
    synthetic_data_params: dict[str, Any],
    client: object,
) -> tuple[str, str]:
    if synthetic_data_params["data_file"] is None:
        instructional_prompt: str = f"Pretend you are a human interacting with a customer service chatbot for the following company: {summary}\nYou have the following goal when interacting with this chatbot:\n{goal}\nHere is a description of the customer you are pretending to be:\n{profile}\nHave a conversation with the chatbot while trying to achieve your goal as this customer. Make sure the conversation is natural. For example, if the chatbot asks you a question you should answer it."
        start_text: str = "Humans write short questions with occasional typos. Here are some examples of what a human customer would type: [how much is it?, Can you send info to my email, yes I need a job, want to check both proposals to rent and buy, How much does it cost a [PRODUCT_HERE], Im interested in [PRODUCT_HERE], hi i would like to rent out [PRODUCT_HERE] but im wondering which countries are available for rental]. Replicate the writing behavior of a human customer and begin the conversation with a question to achieve your goal."
    else:
        example_convo, matching_profile = get_example_convo(
            attr, synthetic_data_params, summary, client
        )
        instructional_prompt: str = f"Pretend you are a human interacting with a customer service chatbot for the following company: {summary}\nYou have the following goal when interacting with this chatbot:\n{goal}\nHere is a description of the customer you are pretending to be:\n{profile}\nHave a conversation with the chatbot while trying to achieve your goal as this customer. Make sure the conversation is natural. For example, if the chatbot asks you a question you should answer it. Below is an example conversation between a user with a similar profile to yours that you can use a guide. However, keep in mind that the users profile may not be the exact same as yours, so take that into consideration when conducting the conversation. Here is the sample users profile:\n{matching_profile}\nAnd here is the conversation between this user and the chatbot:\n{example_convo}"
        start_text: str = "Replicate the writing behavior of a human customer and begin the conversation with a question to achieve your goal."
    return instructional_prompt, start_text


def check_goal_completion(
    goal: str, convo: list[dict[str, Any]], client: object
) -> bool:
    convo_str: str = format_chat_history_str(flip_hist_content_only(convo[2:]))
    prompt: str = f"Here is a conversation between a user and a customer service chatbot assistant:\n{convo_str}\n\nThe user's goal is the following: {goal}\nOutput False if the user needs to learn more information regarding their goal. Output True otherwise. Only onput True or False and nothing else."
    output: str = chatgpt_chatbot([{"role": "user", "content": prompt}], client)
    return output == "True"


def conversation(
    model_api: str,
    profile: str,
    goal: str,
    attr: dict[str, Any],
    sys_input: dict[str, Any],
    summary: str,
    model_params: dict[str, Any],
    synthetic_data_params: dict[str, Any],
    env_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], bool]:
    instructional_prompt, start_text = retrieve_prompts(
        profile, goal, attr, summary, synthetic_data_params, env_config["client"]
    )
    history: list[dict[str, Any]] = []
    history.append({"role": "system", "content": instructional_prompt})
    history.append({"role": "user", "content": start_text})
    chatbot_history: list[dict[str, Any]] = []
    default_slots: list[dict[str, Any]] = []
    for key, value in sys_input.items():
        if key and value:
            default_slots.append({"name": key, "value": value})
    model_params = {"taskgraph": {"dialog_states": {"default_slots": default_slots}}}
    goal_completetion: bool = False

    for i in range(synthetic_data_params["max_turns"]):
        output: str = chatgpt_chatbot(history, env_config["client"])
        history.append({"role": "assistant", "content": output})
        chatbot_history.append({"role": "assistant", "content": output})
        response_data: dict[str, Any] = query_chatbot(
            model_api, chatbot_history, model_params, env_config
        )
        answer: str = response_data["answer"]
        answer = answer.replace("\n", " ")
        model_params = response_data["parameters"]
        history[-1]["intent"] = model_params["taskgraph"]["curr_global_intent"]
        history[-1]["curr_node"] = model_params["taskgraph"]["curr_node"]
        history[-1]["trajectory"] = model_params["memory"]["trajectory"][-1]

        history.append({"role": "user", "content": answer})
        chatbot_history.append({"role": "user", "content": answer})
        if i > 2 and check_goal_completion(goal, history.copy(), env_config["client"]):
            goal_completetion = True
            # history.append({'goal_completetion': True})
            break

    # if not history[-1].get('goal_completetion', False):
    #     history.append({'goal_completetion': False})
    return history, goal_completetion


def generate_conversations(
    model_api: str,
    profiles: list[str],
    goals: list[str],
    attributes_list: list[dict[str, Any]],
    system_inputs: list[dict[str, Any]],
    summary: str,
    model_params: dict[str, Any],
    synthetic_data_params: dict[str, Any],
    env_config: dict[str, Any],
) -> list[dict[str, Any]]:
    convos: list[dict[str, Any]] = []

    # Create input combinations with IDs
    input_combinations = []
    for i, (profile, goal, attr, sys_input) in enumerate(
        zip(profiles, goals, attributes_list, system_inputs, strict=False)
    ):
        input_combinations.append(
            {
                "id": i,
                "profile": profile,
                "goal": goal,
                "attr": attr,
                "sys_input": sys_input,
            }
        )

    print(
        f"Starting conversation generation for {len(input_combinations)} combinations..."
    )

    def worker(input_combo: dict[str, Any]) -> dict[str, Any]:
        print(
            f"Processing conversation {input_combo['id'] + 1}/{len(input_combinations)}"
        )
        convo, goal_completion = conversation(
            model_api,
            input_combo["profile"],
            input_combo["goal"],
            input_combo["attr"],
            input_combo["sys_input"],
            summary,
            model_params,
            synthetic_data_params,
            env_config,
        )
        syn_convo = flip_hist(filter_convo(convo, filter_turns=False))
        print(
            f"Completed conversation {input_combo['id'] + 1} (goal completion: {goal_completion})"
        )
        return {
            "id": input_combo["id"],
            "convo": syn_convo,
            "profile": input_combo["profile"],
            "goal": input_combo["goal"],
            "attributes": input_combo["attr"],
            "goal_completion": goal_completion,
        }

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(worker, input_combo) for input_combo in input_combinations
        ]

        # Store results in a list with the same length as input
        results = [None] * len(input_combinations)
        for future in as_completed(futures):
            result = future.result()
            results[result["id"]] = result

        # Filter out None values and convert to final format
        convos = [r for r in results if r is not None]

    print(f"Conversation generation completed. Generated {len(convos)} conversations.")
    return convos


def simulate_conversations(
    model_api: str,
    model_params: dict[str, Any],
    synthetic_data_params: dict[str, Any],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    print(f"Starting simulation with task: {config['task']}")

    if config["task"] == "first_pass" or config["task"] == "all":
        print("Building user profiles...")
        profiles, goals, attributes_list, system_inputs, labels_list = build_profile(
            synthetic_data_params, config
        )
        print(f"Built {len(profiles)} profiles, {len(goals)} goals")

        # save the profiles, goals, attributes_list, system_inputs, labels_list in a json file
        print("Saving profile data to files...")
        os.makedirs(os.path.join(config["output_dir"], "simulate_data"), exist_ok=True)
        with open(
            os.path.join(config["output_dir"], "simulate_data", "profiles.json"), "w"
        ) as f:
            json.dump(profiles, f, indent=4)
        with open(
            os.path.join(config["output_dir"], "simulate_data", "goals.json"), "w"
        ) as f:
            json.dump(goals, f, indent=4)
        with open(
            os.path.join(config["output_dir"], "simulate_data", "attributes_list.json"),
            "w",
        ) as f:
            json.dump(attributes_list, f, indent=4)
        with open(
            os.path.join(config["output_dir"], "simulate_data", "system_inputs.json"),
            "w",
        ) as f:
            json.dump(system_inputs, f, indent=4)
        with open(
            os.path.join(config["output_dir"], "simulate_data", "labels_list.json"), "w"
        ) as f:
            json.dump(labels_list, f, indent=4)
        print("Profile data saved successfully.")

    elif config["task"] == "simulate_conv_only":
        print("Loading existing profile data...")
        with open(
            os.path.join(config["output_dir"], "simulate_data", "profiles.json")
        ) as f:
            profiles = json.load(f)
        with open(
            os.path.join(config["output_dir"], "simulate_data", "goals.json")
        ) as f:
            goals = json.load(f)
        with open(
            os.path.join(config["output_dir"], "simulate_data", "attributes_list.json"),
        ) as f:
            attributes_list = json.load(f)
        with open(
            os.path.join(config["output_dir"], "simulate_data", "system_inputs.json"),
        ) as f:
            system_inputs = json.load(f)
        with open(
            os.path.join(config["output_dir"], "simulate_data", "labels_list.json")
        ) as f:
            labels_list = json.load(f)
        print("Profile data loaded successfully.")

    summary: str = config["intro"]
    env_config: dict[str, Any] = {
        "workers": config["workers"],
        "tools": config["tools"],
        "client": config["client"],
    }

    print("Starting conversation generation...")
    conversations: list[dict[str, Any]] = generate_conversations(
        model_api,
        profiles,
        goals,
        attributes_list,
        system_inputs,
        summary,
        model_params,
        synthetic_data_params,
        env_config,
    )
    print(f"Simulation completed. Generated {len(conversations)} conversations.")
    return conversations, goals


def main(
    model_api: str = "http://adaptation.cs.columbia.edu:55231/qa/richtech/v1alpha1",
    synthetic_data_params: dict[str, Any] | None = None,
    model_params: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    output_file: str = "p1_sample_convos.json",
) -> list[dict[str, Any]]:
    """
    Main function to simulate conversations with configurable parameters.

    Args:
        model_api: The API endpoint for the model
        synthetic_data_params: Parameters for synthetic data generation
        model_params: Parameters for the model
        config: Configuration dictionary for the simulation
        output_file: Output file path for saving conversations

    Returns:
        List of conversation dictionaries
    """
    # Set default values if not provided
    if synthetic_data_params is None:
        synthetic_data_params = {
            "num_convos": 5,
            "num_goals": 3,
            "max_turns": 10,
        }

    if model_params is None:
        model_params = {}

    if config is None:
        config = {
            "task": "first_pass",
            "output_dir": ".",
            "intro": "Sample company introduction",
            "workers": {},
            "tools": {},
            "client": None,
        }

    # Simulate conversations
    convos, _ = simulate_conversations(
        model_api, model_params, synthetic_data_params, config
    )

    # Save to file if output_file is provided
    if output_file:
        with open(output_file, "w") as f:
            json.dump(convos, f, indent=5)

    return convos


if __name__ == "__main__":
    convos = main()
