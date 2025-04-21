import os
import json
import argparse

from arklex.evaluation.simulate_first_pass_convos import simulate_conversations
from arklex.evaluation.extract_conversation_info import extract_task_completion_metrics
from arklex.evaluation.simulate_second_pass_convos import get_labeled_convos
from arklex.utils.model_config import MODEL
from arklex.utils.model_provider_config import LLM_PROVIDERS
from arklex.evaluation.chatgpt_utils import create_client

def evaluate(config):
    task = config['task']
    model_api = config['model_api']
    model_params = config['model_params']
    synthetic_data_params = config['synthetic_data_params']
    bot_goal = config.get('builder_objective', None)
    bot_goal = None if bot_goal == "" else bot_goal
    
    if task == 'first_pass':
        # first pass
        first_pass_data, goals = simulate_conversations(model_api, model_params, synthetic_data_params, config)
        goal_metrics = extract_task_completion_metrics(first_pass_data, config['client'], bot_goal)
        data = first_pass_data
    
    # second pass
    if task == 'all':
        labeled_convos = get_labeled_convos(first_pass_data, model_api, synthetic_data_params, model_params, config)
    else:
        labeled_convos = []
    return data, labeled_convos, goal_metrics, goals


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_api', type=str)
    parser.add_argument('--model_params', type=dict, default={})
    parser.add_argument('--num_convos', type=int, default=5)
    parser.add_argument('--num_goals', type=int, default=5)
    parser.add_argument('--max_turns', type=int, default=5)
    parser.add_argument('--documents_dir', type=str)
    parser.add_argument('--config', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--model', type=str, default=MODEL["model_type_or_path"])
    parser.add_argument( '--llm-provider',type=str,default=MODEL["llm_provider"],choices=LLM_PROVIDERS)
    parser.add_argument('--customer_type', type=str, default=None, choices=['b2b', 'b2c'])
    parser.add_argument('--task', type=str, default='first_pass', choices=['first_pass', 'all'])
    parser.add_argument('--user_attributes', type=str, default='arklex/evaluation/user_attributes.json')
    parser.add_argument('--custom_profile', action='store_true')
    parser.add_argument('--system_inputs', action='store_true')
    parser.add_argument('--data_file', type=str, default=None)
    args = parser.parse_args()

    MODEL["model_type_or_path"] = args.model
    MODEL["llm_provider"] = args.llm_provider
    client = create_client()

    assert args.model_api is not None, "Model api must be provided"
    assert args.config is not None, "Config file must be provided"
    assert args.user_attributes is not None, "User attribute file must be provided"
    if not args.output_dir:
        args.output_dir = os.path.join(args.documents_dir, 'eval')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    config = json.load(open(args.config))
    user_attributes = json.load(open(args.user_attributes))
    # if args.testset:
    #     testset = json.load(open(args.testset))
    # else:
    #     testset = {}
    config['model_api'] = args.model_api
    config['documents_dir'] = args.documents_dir
    config['output_dir'] = args.output_dir
    config['model_params'] = args.model_params
    config['synthetic_data_params'] = {'num_convos': args.num_convos, 'num_goals': args.num_goals, 
                                       'max_turns': args.max_turns, 'customer_type': args.customer_type, 
                                       'data_file': args.data_file}
    config['task'] = args.task
    config['user_attributes'] = user_attributes
    config['custom_profile'] = args.custom_profile
    config['system_inputs'] = args.system_inputs
    config['client'] =client
    first_pass_data, final_convos, goal_metrics, goals = evaluate(config)

    with open(os.path.join(args.output_dir, 'goals.json'), 'w') as f:
        json.dump(goals, f, indent=4)

    with open(os.path.join(args.output_dir, 'simulate_data.json'), 'w') as f:
        json.dump(first_pass_data, f, indent=4)

    with open(os.path.join(args.output_dir, 'labeled_data.json'), 'w') as f:
        json.dump(final_convos, f, indent=4)
    
    with open(os.path.join(args.output_dir, 'goal_completion.json'), 'w') as f:
        json.dump(goal_metrics, f, indent=4)