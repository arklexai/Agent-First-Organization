import json
import networkx as nx

from arklex.evaluation.chatgpt_utils import chatgpt_chatbot, format_chat_history_str, flip_hist_content_only, filter_convo

def get_edges_and_counts(data):
    edge_counts = {}
    for convo in data:
        convo = filter_convo(convo)
        for i in range(len(convo)):
            if convo[i]['role'] == 'assistant':
                continue
            prev_intent = 'start' if i == 0 else convo[i-2]['intent']
            current_intent = convo[i]['intent']
            edge_counts[(prev_intent, current_intent)] = edge_counts.get((prev_intent, current_intent), 0) + 1
    return edge_counts

def build_intent_graph(data):
    G = nx.DiGraph()
    edge_counts = get_edges_and_counts(data)
    for key in edge_counts.keys():
        G.add_edge(key[0], key[1], weight = edge_counts[key])
    return G

def check_bot_goal(convo, bot_goal, client):
    convo_str = format_chat_history_str(flip_hist_content_only(convo))
    prompt = f"Here is a conversation between a user and a customer service chatbot assistant:\n{convo_str}\n\nThe chatbot's goal is the following: {bot_goal}\nOutput True if the bot was able to achieve its goal. Output False otherwise. Only output True or False and nothing else."
    output = chatgpt_chatbot([{'role': 'user', 'content': prompt}], client)
    return output == "True"

def num_user_turns(convo):
    user_turns = 0
    for turn in convo:
        if turn.get('role', None) == 'user':
            user_turns += 1
    return user_turns

def extract_task_completion_metrics(data, client, bot_goal=None):
    num_convos = len(data)
    if num_convos == 0:
        return "Error while extracting task completion metrics"
    goal_completetions = 0
    bot_goal_completions = 0
    completion_efficiency = 0
    for convo in data:
        convo_history = convo['convo']
        completion_efficiency += num_user_turns(convo_history)
        if convo["goal_completion"]:
            goal_completetions += 1
        if bot_goal is not None and check_bot_goal(convo_history, bot_goal, client):
            bot_goal_completions += 1
    metrics = {'user_task_completion': goal_completetions/num_convos,
               'user_task_completion_efficiency': completion_efficiency/num_convos}
    if bot_goal is not None:
        metrics['bot_goal_completion'] = bot_goal_completions/num_convos
    return metrics

if __name__ == "__main__":
    # with open('files/p1_sample_convos.json') as f:
    #     data = json.load(f)

    # model_api = "http://adaptation.cs.columbia.edu:55131/predict"
    # model_params = {'bot_id' : 'richtech', 'bot_version': 'v1alpha1'}
    # convos  = get_nlu_labels(data, model_api, model_params)
    # with open('files/p1_sample_convos_labeled.json', 'w') as f:
    #     json.dump(convos, f, indent=5)

    
    with open('files/p1_sample_convos_labeled.json') as f:
        data = json.load(f)
    G = build_intent_graph(data)
    for e in list(G.edges()):
        print(f"Weight for edge {e}: {G.get_edge_data(e[0], e[1])['weight']}")
