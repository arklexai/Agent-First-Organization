import os
import random
import json
import requests
from openai import OpenAI
import anthropic
from dotenv import load_dotenv

from arklex.utils.model_config import MODEL
load_dotenv()


def create_client():
    try:
        org_key = os.environ["OPENAI_ORG_ID"]
    except:
        org_key = None
    if MODEL['llm_provider'] == 'openai' or MODEL['llm_provider']== 'gemini':
        client = OpenAI(
            api_key=os.environ[f"{MODEL['llm_provider'].upper()}_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/" if MODEL['llm_provider'] == 'gemini' else None,
            organization=org_key
        )
    elif MODEL['llm_provider'] == 'anthropic':
        client = anthropic.Anthropic()
    return client
    

def chatgpt_chatbot(messages, client, model=MODEL["model_type_or_path"]):
   
    if MODEL['llm_provider'] != 'anthropic':
        answer = client.chat.completions.create(
            model=MODEL['model_type_or_path'], messages=messages, temperature=0.1
        ).choices[0].message.content.strip()
    else:
        kwargs = {
            "model": MODEL["model_type_or_path"],
            "messages": messages if messages[0]['role'] != 'system' else [messages[1]],
            "temperature": 0.1,
            "max_tokens": 1024,
            **({"system": messages[0]['content']} if messages[0]['role'] == 'system' else {})
        }
        answer = client.messages.create(**kwargs).content[0].text.strip()
   
    return answer

# flip roles in convo history, only keep role and content
def flip_hist_content_only(hist):
    new_hist = []
    for turn in hist:
        if turn['role'] == 'system':
            continue
        elif turn['role'] == 'user':
            new_hist.append({'role': 'assistant', 'content': turn['content']})
        else:
            new_hist.append({'role': 'user', 'content': turn['content']})
    return new_hist

# flip roles in convo history, keep all other keys the same
def flip_hist(hist):
    new_hist = []
    for turn in hist.copy():
        if 'role' not in turn.keys():
            new_hist.append(turn)
        elif turn['role'] == 'system':
            continue
        elif turn['role'] == 'user':
            turn['role'] = 'assistant'
            new_hist.append(turn)
        else:
            turn['role'] = 'user'
            new_hist.append(turn)
    return new_hist

def query_chatbot(model_api, history, params, env_config):
    history = flip_hist_content_only(history)
    data = {
        "history": history,
        "parameters": params,
        "workers": env_config["workers"],
        "tools": env_config["tools"],
    }
    data = json.dumps(data)
    response = requests.post(model_api, headers={"Content-Type": "application/json"}, data=data)
    return response.json()

def format_chat_history_str(chat_history):
    formatted_hist = ''
    for turn in chat_history:
        formatted_hist += turn['role'].upper() + ': ' + turn['content'] + ' '
    return formatted_hist.strip()

# filter prompts out of bot utterances
def filter_convo(convo, delim = '\n', filter_turns = True):
    filtered_convo = []
    for i, turn in enumerate(convo):
        if i <= 1:
            continue
        elif 'role' not in turn.keys() and filter_turns:
            continue
        elif 'role' not in turn.keys():
            filtered_convo.append(turn)
        elif turn['role'] == 'assistant':
            filtered_convo.append(turn)
        else:
            idx = turn['content'].find(delim)
            new_turn = {}
            for key in turn.keys():
                if key == 'content':
                    new_turn[key] = turn[key][:idx]
                else:
                    new_turn[key] = turn[key]
            filtered_convo.append(new_turn)
    return filtered_convo

def adjust_goal(doc_content, goal):
    message = f"Pretend you have the following goal in the mind. If the goal including some specific product, such as floss, mug, iphone, etc., then please replace it with the product from the following document content. Otherwise, don't need to change it and just return the original goal. The document content is as follows:\n{doc_content}\n\nThe original goal is as follows:\n{goal}\n\nOnly give the answer to the question in your response."

    return chatgpt_chatbot([{'role': 'user', 'content': message}], model=MODEL["model_type_or_path"])

def generate_goal(doc_content, client):
    message = f"Pretend you have just read the following website:\n{doc_content}\nThis website also has a chatbot. What is some information you want to get from this chatbot or a goal you might have when chatting with this chatbot based on the website content? Answer the question in the first person. Only give the answer to the question in your response."
    
    return chatgpt_chatbot([{'role': 'user', 'content': message}], client, model=MODEL["model_type_or_path"])

def generate_goals(documents, params,client):
    goals = []
    for i in range(params['num_goals']):
        doc = random.choice(documents)
        goals.append(generate_goal(doc['content']), client)
    return goals

if __name__ == '__main__':
    from get_documents import get_all_documents, filter_documents
    documents = get_all_documents()
    documents = filter_documents(documents)
    client = create_client()
    params = {'num_goals': 1}
    print(generate_goals(documents, params, client))