# User Simulator

In order to evaluate the performance of the chatbot, we will simulate user interactions with the chatbot. The user simulator consist of **two-pass** process:

1. Simulate user's utterances based on the config file and interact with the built chatbot to get the task success rate based on the completion of user's goal and builder's goal.
2. Simulate user's utterances based on the generated taskgraph and interact with the built chatbot to get the intent prediction accuracy to evaluate the NLU performance.

The user simulator can be used to generate a large number of user inputs to evaluate the chatbot's performance under different scenarios.

## Parameters

The following parameters control the generation of synthetic conversations:

* `--num_convos`: Number of synthetic conversations to simulate. Default is 5.
* `--num_goals`: Number of goals/tasks to simulate. Default is 5.
* `--max_turns`: Maximum number of turns per conversation. Default is 5.
* `--model`: The openai model type used to synthesize user's utterance. Default is `gpt-4o`. You could change it to other models like `gpt-4o-mini`.

> **Note:** `--max_turns` will affect the **User's goal completion efficiency** metrics since if the maximum number of turns is set to be small (e.g 2, 3), the chatbot may not be able to complete the user's goal and the efficiency will be `--max_turns` which is not the actual efficiency. Normally, it should be set to a larger number like 5 - 10.
