import os
import argparse
import json
from typing import List, Dict, Any
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

RELEVANT_GOALS = [
    "order tracking",
    "asking for refund",
    "report order missing",
    "report product issue",
    "asking replacement",
    "request repair",
    "interested in product information",
    "product purchase",
    "product rental",
]
CHECK_ENGLISH = "Is this conversation in English:\n{conversation}"
CHECK_GOAL = "Here is a conversation between a user and a customer service chatbot:\n{conversation}\nGive a guess as to what the user's goal was during this conversation, explain your reasoning. After that, determine if the goal you picked falls into one of the categories below, explain your reasoning followed by your final answer which should be either 'yes' or 'no'. If you have any uncertanties about the user's goal your answer should be 'no'. Here are the goal categories:\n{goals}"

SUMMARY = "Richtech Robotics's headquarter is in Las Vegas; the other office is in Austin. Richtech Robotics provide worker robots (ADAM, ARM, ACE), delivery robots (Matradee, Matradee X, Matradee L, Richie), cleaning robots (DUST-E SX, DUST-E MX) and multipurpose robots (skylark). Their products are intended for business purposes, but not for home purpose; the ADAM robot is available for purchase and rental for multiple purposes. This robot bartender makes tea, coffee and cocktails. Richtech Robotics also operate the world's first robot milk tea shop, ClouTea, in Las Vegas (www.cloutea.com), where all milk tea beverages are prepared by the ADAM robot. The delivery time will be one month for the delivery robot, 2 weeks for standard ADAM, and two months for commercial cleaning robot."
CHECK_CUSTOMER = "Here is a conversation between a human and a customer service chatbot:\n{convo}\nBased on this conversation, do you think that this person could be a potential customer for the company. Explain your reasoning followed by your final answer of yes or no. Here is a summary of the company to help you decide:\n{summary}"
CHECK_ENGAGEMENT = "Here is a conversation between a human and a customer service chatbot:\n{convo}\nBased on this conversation assess whether the user is positively engaged in the conversation and with the chatbot. Answer yes if the user is engaged and no if the user is not engaged. An example of an unengaged user may be one who gives short replies or asks the bot questions that aren't relevant to customer service. Err on the side on answering no unless the user is highly engaged in the conversation. We will be reaching out to these users individually later so don't want to expend the effort unless they were very engaged with the chatbot. Explain your reasoning about whether or not the user is engaged followed by your final answer of yes or no."

FULL_PROFILE = """
goal: order tracking, asking for refund, report order missing, report product issue, asking replacement, request repair, interested in product information, product purchase, product rental, other
product_experience_level: new to this product, used competitor product, other
deal_stage: new leads, demo schedule, after demo, decision make buy in, contract sent, contract signed, money paid, other
customer_type: new prospect, returning customer, other
decision_making_authority: primary, secondary, other
persona: aggressive, passive, neutral, other
discovery_type: social media, friend recommendation, search engine results, third party website, other
buying_behavior: economic, quality oriented, other
"""
GET_CUSTOMER_PROFILE = "Your job is to take conversations between a user and a chatbot then use it to build a customer profile for the user. Each profile contains a user attribute followed by a value for that attribute. Here is a list of all the attributes followed by the posisble values for that attribute:{full_profile}Based on the conversation below, create a user profile for the user. The profile should contain each attribute followed by exactly one of the values given above, if you're not sure what the value for an attribute should be just put 'other'. Explain your reasoning for each attribute followed by the final profile. It should look exactly like the one given above except for each attribute should only have one value. Here is the conversation:\n{convo}\nYour final output should be formatted as follows:\n[REASONING_FOR_EACH_ATTRIBUTE]\nFinal Profile:\n[FINAL_PROFILE_TEXT]\nPut the output in plain text, there should be no markdown or other formatting like that."


client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"], organization=os.environ["OPENAI_ORG_ID"]
)


def chatgpt_chatbot(messages: List[Dict[str, str]], model: str) -> str:
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    answer: str = completion.choices[0].message.content.strip()
    return answer


def join_messages(messages: List[Dict[str, str]]) -> str:
    message_str: str = ""
    for message in messages:
        if message["role"] == "bot_follow_up":
            continue
        message_str += f"{message['role']}: {message['content']}\n"
    return message_str[:-1]


def rule_based_filtering(convos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    bad_msg_order: bool = False
    final_convos: List[Dict[str, Any]] = []
    for convo in convos:
        if convo["user_msg_length"] < 5:
            continue
        for i in range(len(convo["message"]) - 1):
            if (
                convo["message"][i]["role"] == "user"
                and convo["message"][i + 1]["role"] == "user"
            ):
                bad_msg_order = True
                break

        if bad_msg_order:
            bad_msg_order = False
            continue
        final_convos.append(convo)

    return final_convos


def first_pass_model_filtering(
    convos: List[Dict[str, Any]], model: str
) -> List[Dict[str, Any]]:
    final_convos: List[Dict[str, Any]] = []
    for convo in tqdm(convos):
        convo_str: str = join_messages(convo["message"])
        english_check: str = chatgpt_chatbot(
            [{"role": "user", "content": CHECK_ENGLISH.format(conversation=convo_str)}],
            model=model,
        )
        if "no" in english_check:
            continue

        goal_guess: str = chatgpt_chatbot(
            [
                {
                    "role": "user",
                    "content": CHECK_GOAL.format(
                        conversation=convo_str, goals=RELEVANT_GOALS
                    ),
                }
            ],
            model=model,
        )
        final_answer: str = goal_guess.split("\n")[-1]
        if "yes" in final_answer:
            final_convos.append(convo)

    return final_convos


def second_pass_model_filtering(
    convos: List[Dict[str, Any]], model: str
) -> List[Dict[str, Any]]:
    final_convos: List[Dict[str, Any]] = []
    for convo in tqdm(convos):
        convo_str: str = join_messages(convo["message"])
        engagement_guess: str = chatgpt_chatbot(
            [{"role": "user", "content": CHECK_ENGAGEMENT.format(convo=convo_str)}],
            model=model,
        )
        final_answer: str = engagement_guess.split("\n")[-1]
        if "yes" in final_answer:
            final_convos.append(convo)
    return final_convos


def extract_customer_profile(customer_profile_output: str) -> Dict[str, str]:
    split_profile: List[str] = customer_profile_output.split("Final Profile:\n")
    profile_text: str = split_profile[-1]
    profile: Dict[str, str] = {}
    for line in profile_text.split("\n"):
        if ":" not in line:
            continue
        key: str
        val: str
        key, val = line.split(": ")
        profile[key] = val
    return profile


def get_all_customer_profiles(
    final_convos: List[Dict[str, Any]], model: str
) -> Dict[str, List[Dict[str, Any]]]:
    failed_extractions: int = 0
    profiles: Dict[str, List[Dict[str, Any]]] = {}
    for convo in tqdm(final_convos):
        convo_str: str = join_messages(convo["message"])
        customer_profile_output: str = chatgpt_chatbot(
            [
                {
                    "role": "user",
                    "content": GET_CUSTOMER_PROFILE.format(
                        full_profile=FULL_PROFILE, convo=convo_str
                    ),
                }
            ],
            model=model,
        )
        try:
            profile: Dict[str, str] = extract_customer_profile(customer_profile_output)
        except:
            failed_extractions += 1
            print(f"Profile extraction failed. Total failures: {failed_extractions}")
        profile_vals: str = ",".join(list(profile.values()))
        profiles[profile_vals] = profiles.get(profile_vals, []) + [convo]
    return profiles


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_file", type=str)
    parser.add_argument("--filtering_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--extraction_model", type=str, default="gpt-4o")
    parser.add_argument("--output_data_file", type=str, default="profiles.json")
    args = parser.parse_args()

    with open(args.data_file) as f:
        convos = json.load(f)

    rule_filtered_convos = rule_based_filtering(convos)
    model_filtered_convos = first_pass_model_filtering(convos, args.filtering_model)
    filtered_convos = second_pass_model_filtering(convos, args.filtering_model)
    profiles = get_all_customer_profiles(filtered_convos, args.extraction_model)

    with open(args.output_data_file, "w") as f:
        json.dump(profiles, f, indent=4)
