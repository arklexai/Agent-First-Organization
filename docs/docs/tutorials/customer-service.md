# Customer Service agent

*Enhance your agents with Retrieval Augmented Generation (RAG)*

Follow the video tutorial here:

<iframe
width="100%"
height="440"
src="https://www.youtube.com/embed/y1P2Ethvy0I"
title="YouTube video player"
frameborder="0"
allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
allowfullscreen></iframe>

## Intro

In this tutorial, we'll walk through building a basic customer service agent using **Arklex Agent First Organization**'s framework. This agent will be able to handle common customer inquiries, such as answering FAQs, identifying customer preferences, and requesting customer's contact information.

By the end of this tutorial, you'll know how to set up the AI framework, prepare documents for RAG, build a basic conversation flow, and power an agent with it. Whether you're building your first agent or refining your skills, this tutorial will guide you in creating a responsive and helpful customer service agent.

## Supplying Information

Before any technical work is done, we must first identify and find information to supply to the agent. After all, you can't serve as a Customer Service Agent without anything to support! To add resources to the agent, we need to decide what the agent should know and look for links to sources in which the agent can learn from. E.g, a bunch of Wikipedia content. In this example, we will be using a company's website to supply to the agent. Keep it saved somewhere, we will be needing it for our next step!

## Setting up the Config File

At its core, the agent is powered through a *TaskGraph* which is the structure that links various tasks together to fulfill the overall role of the agent. Each "node" represents a task that has a *Worker* that is selected to complete the task. Each node engages the user for their response, and  with the user response, the TaskGraph will decide which next node to travel to.

Like actual conversations, *TaskGraph* can be complicated; that is why we help you convert a simple and intuitive *Config* JSON file into a powerful and advanced *TaskGraph* through our generator. Instead of needing to design an entire graph, all you need to do is to describe the agent and provide some extra information and it will build the graph for you!

#### Workers

Building on top of a primitive conversational agent, for the customer service agent, we need to let the agent be able to read from the documents we are supplying when composing a response. We can do that through the [RAGWorker](../Workers/RAGWorker.mdx) which retrieves the relevant information from our sources and then passes the information to [MessageWorker](../Workers/MessageWorker.mdx) which composes the response.

As a refresher, here is the structure for a [Config](../Config/intro.md) JSON file:

* `role (Required)`: The general "role" of the agent you want to create. For instance, "customer service assistant", "data analyst", "shopping assistant", etc.
* `user_objective (Required)`: The user's goal that the agent wants to achieve. Related to the user experience. Description in third person. For instance, "The customer service assistant helps users with customer service inquiries. It can provide information about products, services, and policies, as well as help users resolve issues and complete transactions."
* `builder_objective (Optional)`: The additional target you want the agent to achieve beyond the user's goal. Can contain hidden objectives or subtle objectives which is hidden from the user. Describe in the third person. For instance, "The customer service assistant helps to request customer's contact information."
* `domain (Optional)`: The domain that you want to create the agent for. For instance, "robotics and automation", "Ecommerce", "Healthcare", etc.
* `intro (Optional)`: The introduction of the above domain that you want to create the agent for. It should contain the information about the domain, the products, services, and policies, etc.
* `task_docs (Optional, Dict)`: The documents resources for the taskgraph generation to create the chatbot. The dictionary should contain the following fields:
            *`source (Required)`: The source url that you want the chatbot to refer to
            * `desc (Optional)` : Short description of the source and how it is used
            * `num (Optional)`: The number of websites that you want the chatbot to refer to for the source, defaults to one (only the url page)
* `rag_docs (Optional, Dict)`: The documents resources for the rag component of chatbot when running chatbot. The dictionary should contain the following fields:
            *`source (Required)`: The source url that you want the chatbot to refer to
            * `desc (Optional)` : Short description of the source and how it is used
            * `num (Optional)`: The number of websites that you want the chatbot to refer to for the source, defaults to one (only the url page)
* `tasks (Optional, List(Str))`: The pre-defined list of tasks that the chatbot need to handle. If empty, the system will generate the tasks and the steps to complete the tasks based on the role, objective, domain, intro and docs fields. The more information you provide in the fields, the more accurate the tasks and steps will be generated.
* `workers (Required, List(Dict))`: The workers pre-defined under arklex/env/workers folder that you want to use for the chatbot. Each worker will be defined as a class decorated with @register_worker. Please refer to the arklex/env/workers/message_worker.py for an example. The field required for each worker object is:
            *`id (Required, uuid)`: The unique id for the worker
            * `name (Required, Str)`: The WorkerClassName. Such as `MessageWorker`
            * `path (Required, Str)`: The file path of the worker start from the arklex/env/workers folder. Such as `message_worker.py`.
* `tools (Optional, List(Dict))`: The tools (e.g. APIs, function, etc.) pre-defined under arklex/env/tools folder that you want to use for the chatbot. Each tool will be defined as a function decorated with @register_tool. The decorator includes the **description** - the purpose of the function, **slots** - the arguments needed for the function, **outputs** - expected result of the function. For more details, please refer to the arklex/env/tools/shopify/find_user_id_by_email.py as an example. The field required for each tool object is:
            *`id (Required, uuid)`: The unique id for the worker
            * `name (Required, Str)`: The tool function name. Such as `find_user_id_by_email`.
            *`path (Required, Str)`: The file path of the worker start from the arklex/env/tools folder. Such as `shopify/find_user_id_by_email.py`.
            * `fixed_args (Optional, Dict)`: All the must and deterministic arguments for the tool function, such as credentials or already known argument during development. It should be a dictionary. Such as `{"token": "<access_token>", "shop_url": "<url>", "api_version": "<version>"}`

```json
{
    "role": "customer service assistant",
    "user_objective": "The customer service assistant helps users with customer service inquiries. It can provide information about products, services, and policies, as well as help users resolve issues and complete transactions.",
    "builder_objective": "The customer service assistant helps to request customer's contact information.",
    "domain": "robotics and automation",
    "intro": "Richtech Robotics's headquarter is in Las Vegas; the other office is in Austin. Richtech Robotics provide worker robots (ADAM, ARM, ACE), delivery robots (Matradee, Matradee X, Matradee L, Richie), cleaning robots (DUST-E SX, DUST-E MX) and multipurpose robots (skylark). Their products are intended for business purposes, but not for home purpose; the ADAM robot is available for purchase and rental for multiple purposes. This robot bartender makes tea, coffee and cocktails. Richtech Robotics also operate the world's first robot milk tea shop, ClouTea, in Las Vegas (www.cloutea.com), where all milk tea beverages are prepared by the ADAM robot. The delivery time will be one month for the delivery robot, 2 weeks for standard ADAM, and two months for commercial cleaning robot. ",
    "task_docs": [{
        "source": "https://www.richtechrobotics.com/",
        "num": 20
    }],
    "rag_docs": [{
        "source": "https://www.richtechrobotics.com/",
        "num": 20
    }],
    "tasks": [],
    "workers": [
        {"id": "FaissRAGWorker", "name": "FaissRAGWorker", "path": "faiss_rag_worker.py"},
        {"id": "26bb6634-3bee-417d-ad75-23269ac17bc3", "name": "MessageWorker", "path": "message_worker.py"},
        {"id": "9c15af81-04b3-443e-be04-a3522124b905", "name": "SearchWorker", "path": "search_worker.py"},
        {"id": "b06c2b58-12c5-41fe-9838-e93b530e42e8", "name": "DefaultWorker", "path": "default_worker.py"}
    ],
    "tools": []
}
```

With our Config in place, the vast majority of work is surprisingly already done! The rest is simply bringing the agent to life.

## Generating a TaskGraph

Now that we have a Config file, generating the graph is the easy part. All you need to do is run

`python create.py --config ./examples/customer_service_config.json --output-dir ./examples/customer_service`

It will first enter into a *task planning* interactive panel where you can see the generated tasks this agent will handle and the following steps to complete the specific tasks. You can also modify the tasks and steps as needed. Once you are satisfied with the result, you could press `s` to save the *task planning* file then it will further generate the final *TaskGraph* file.

TaskGraph provides the graph that the agent will traverse through during the conversation. It provides a guideline for the conversation to make it more controllable and reliable. The details can be viewed at [here](../Taskgraph/Generation.md).

It will also prepare the documents for the RAG component of the agent. It will first crawl the content of the website specified by the `rag_docs` in the config file and saved the documents into `documents.pkl` file. Later on, during the conversation, the agent will execute RAG Worker to retrieve relevent information of user's query from the documents to compose responses to the user.

## Running the agent

With the TaskGraph in place, we can run the agent with the following command:

`python run.py --input-dir ./examples/customer_service`

It will initialize the service (e.g. NLU apis) you need to start the agent and you can start interacting with it!

---

## Sample Conversation
>
> **ASSISTANT**: Hello! Welcome to our Customer Service Assistant. I'm here to provide you with information about our products, services, and policies. How can I assist you today?
>
> **USER**: What products do you offer?
>
> **ASSISTANT**:\
> We offer a range of innovative robots designed for business purposes. Our lineup includes:
>
> 1. **Worker robots:**
>    * ADAM: A versatile robot capable of tasks like food and beverage preparation.
>    * ARM and ACE: Designed for various operational needs.
>
> 2. **Delivery robots:**
>    * Matradee, Matradee X, Matradee L, and Richie: These robots are perfect for automating delivery tasks.
>
> 3. **Cleaning robots:**
>    * DUST-E SX and DUST-E MX: Ideal for maintaining cleanliness in commercial spaces.
>
> 4. **Multipurpose robot:**
>    * Skylark: This robot can handle multiple tasks including easy ordering, vending, and cleaning operations.
>
> If you have any specific questions about a product or need further assistance, feel free to ask!
>
> **USER**: I am interested in Cleaning robots, can you tell me more about the differences between DUST-E SX and DUST-E MX?
>
> **ASSISTANT**:\
> Certainly! the DUST-E SX and DUST-E MX are designed for professional cleaning, but they cater to different needs:
>
> 1. **DUST-E SX:**
>    * Designed for smaller environments.
>    * It's a compact solution for medium to small-sized buildings.
>    * Offers the same high-quality cleaning as the MX but in a smaller size, making it ideal for tighter or more confined spaces.
>
> 2. **DUST-E MX:**
>    * Ideal for larger environments with tight chokepoints.
>    * Cleans at a rate of approximately 22,000 square feet per hour.
>    * Equipped with 40+ sensors for safety, providing a 360° view to handle unexpected obstacles.
>    * Features an easy-to-use touchscreen for operation and convenient dirty water disposal.
>
> DUST-E SX models provide excellent consistency and professional-grade cleaning. If you need more specific details or wish to book a demo, feel free to let me know!
>
> **USER**: DUST-E SX seems more aligned with our needs, can we book a demo?
>
> **ASSISTANT**: Certainly! You can book a demo for the DUST-E SX by visiting the following link: [Book a Demo](https://www.richtechrobotics.com/dust-e-mx#section_contact-form). If you need further assistance or have any questions, feel free to ask!
---

## Evaluation

For a task-oriented dialogue system, you could use the evaluation script to automatically generate synthetic conversations, extracting task completion metrics for evaluating the whole system. For more details, please refer to the [evaluation](../Evaluation/UserSimulator.md) documentation.

1. First, create an API for the Agent you built. It will start an API on the default port 8000.

    ```
    python model_api.py  --input-dir ./examples/customer_service
    ```

2. Then, start the evaluation process:

   ```
    python eval.py \
    --model_api http://127.0.0.1:8000/eval/chat \
    --config ./examples/customer_service_config.json \
    --documents_dir ./examples/customer_service \
    --output_dir ./examples/customer_service
    ```

## Evaluation Results

The evaluation will generate the following outputs in the specified output directory:

1. **Simulated Synthetic Dataset (`simulate_data.json`)**  
   * JSON file containing simulated conversations generated based on the user's objective to evaluate the task success rate.
  
2. **Labeled Synthetic Dataset (`labeled_data.json`)**  
   * JSON file containing labeled conversations generated based on the taskgraph to evaluate the NLU performance.

3. **Goal Completion Metrics (`goal_completion.json`)**  
   * JSON file summarizing task completion statistics based on the agent's ability to achieve specified goals.
