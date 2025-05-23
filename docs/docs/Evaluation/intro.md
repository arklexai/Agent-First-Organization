# Introduction

It is designed to evaluate a task-oriented dialogue system by generating synthetic conversations, extracting task completion metrics, and producing a labeled synthetic dataset.

## Tutorial

Here is an example for the customer service assistant chatbot.

1. First, create an API for the Agent you built. It will start an API on the default port 8000.

    ```
    python model_api.py  --input-dir ./examples/customer_service
    ```

    * Fields:
      * `--input-dir`: The directory that contains the needed files for the orchestrator and documents for the workers.
       * `--llm_provider`: The LLM provider you wish to use.
          * Options: `openai` (default), `gemini`, `anthropic`
      * `--model`: The model type used to generate bot response. The default is `gpt-4o`.
        * You can change this to other models like:
          * `gpt-4o-mini`,  `gemini-2.0-flash` , `claude-3-5-haiku-20241022`
      * `--port`: The port number to start the API. Default is 8000.

2. Then, start the evaluation process:

   ```
    python eval.py \
    --model_api http://127.0.0.1:8000/eval/chat \
    --config ./examples/customer_service_config.json \
    --documents_dir ./examples/customer_service \
    --output-dir ./examples/customer_service
    ```

    * Fields:
      * `--model_api`: The api url that you created in the previous step
      * `--config`: The path to the config file
      * `--documents_dir`: The directory that contains the generated files
      * `--output-dir`: The directory to save the evaluation results
      * `--num_convos`: Number of synthetic conversations to simulate. Default is 5.
      * `--num_goals`: Number of goals/tasks to simulate. Default is 5.
      * `--max_turns`: Maximum number of turns per conversation. Default is 5.
       * `--llm_provider`: The LLM provider you wish to use.
          * Options: `openai` (default), `gemini`, `anthropic`
      * `--model`: The model type used to generate bot response. The default is `gpt-4o`.
        * You can change this to other models like:
          * `gpt-4o-mini`,  `gemini-2.0-flash` , `claude-3-5-haiku-20241022`

## Results

The evaluation will generate the following outputs in the specified output directory:

1. **Simulated Synthetic Dataset (`simulate_data.json`)**  
   * JSON file containing simulated conversations generated based on the user's objective to evaluate the task success rate.
  
2. **Labeled Synthetic Dataset (`labeled_data.json`)**  
   * JSON file containing labeled conversations generated based on the taskgraph to evaluate the NLU performance.

3. **Goal Completion Metrics (`goal_completion.json`)**  
   * JSON file summarizing task completion statistics based on the bot's ability to achieve specified goals.
