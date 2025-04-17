import json
import random
from arklex.evaluation.extract_conversation_info import build_intent_graph
from arklex.evaluation.chatgpt_utils import chatgpt_chatbot, query_chatbot, flip_hist, filter_convo, generate_goals
from arklex.evaluation.get_documents import load_docs

def sampling_paths(start_node, graph, path_length, max_turns, intents):
    children = list(graph.successors(start_node))
    if path_length >= max_turns or len(children) == 0:
        return intents
    weights = []
    for c in children:
        weights.append(graph.get_edge_data(start_node, c)['weight'])
    next_node = random.choices(children, weights)[0]
    intents.append(next_node)
    return sampling_paths(next_node, graph, path_length+1, max_turns, intents)

def get_paths(G, num_paths, max_turns):
    my_paths = []
    for i in range(num_paths):
        my_path = sampling_paths('start', G, 0, max_turns, ['start'])
        my_paths.append(my_path[1:])
    return my_paths

def interact(intent_path, summary, model_api, model_params, client):
    history = []
    instructional_prompt = 'Replicate the behavior of a human customer. You are interacting with customer service chatbot for the following company: ' + summary
    start_text = "Begin the conversation as a human customer with the following intent: " + intent_path[0]
    history.append({'role': 'system','content': instructional_prompt})
    history.append({'role': 'user', 'content': start_text})
    for i in range(len(intent_path)):
        intent = intent_path[i]
        output = chatgpt_chatbot(history, client) 
        history.append({'role': 'assistant', 'content': output, 'intent': intent})
        response_data = query_chatbot(model_api, filter_convo(history), model_params)
        answer = response_data["answer"]
        answer = answer.replace('\n', ' ')
        model_params = response_data.get("parameters", model_params)
        if i < len(intent_path) - 1:
            intent = intent_path[i+1]
        history.append({'role': 'user', 'content': answer + '\nRespond to this utterance with the following intent: ' + intent + '\nMake sure your response is natural and follows the flow of the conversation. For example, if the bot asks you a question make sure you answer it.'})
    return history

def generate_labeled_convos(intent_paths, summary, model_api, model_params, client):
    convos = []
    model_params = {}
    for intent_path in intent_paths:
        convo = interact(intent_path, summary, model_api, model_params, client)
        convos.append(flip_hist(filter_convo(convo)))
    return convos

def get_labeled_convos(first_pass_data, model_api, synthetic_data_params, model_params, config):
    intent_graph = build_intent_graph(first_pass_data)
    intent_paths = get_paths(intent_graph, synthetic_data_params['num_convos'], synthetic_data_params['max_turns'])
    summary, client = config['intro'], config['client']
    convos = generate_labeled_convos(intent_paths, summary, model_api, model_params,client)
    return convos


if __name__ == "__main__":
    with open('temp_files/p1_sample_convos_labeled.json') as f:
        data = json.load(f)
    
    with open('temp_files/richtech_config.json') as f:
        config = json.load(f)

    model_api = "http://adaptation.cs.columbia.edu:55231/qa/richtech/v1alpha1"
    synthetic_data_params = {'num_convos': 2, 'num_goals': 3, 'max_turns': 10}
    model_params = {}

    labeled_convos = get_labeled_convos(data, model_api, synthetic_data_params, model_params, config)

    with open('files/p2_sample_convos.json', 'w') as f:
        json.dump(labeled_convos, f, indent=5)