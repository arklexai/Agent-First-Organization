# Roleplay Bot

*Build a basic chatbot with MessageWorkers*

## Intro

In this tutorial, we'll walk through building a roleplay bot using **Agent First Organization**'s framework. This bot will pretend to be a role you assigned to it and have a practice conversation with you based on the objective you set.

By the end of this tutorial, you'll know how to set up the config file, build a basic conversation flow, and power a bot with it! This tutorial covers the absolute basics of our framework and sets up for much more!

## Setting up the Config File

In its core, bot is powered through a [TaskGraph](../Taskgraph/intro.md) which is the structure that links various tasks together to fulfill the overall role of the bot. Every "node" represents a task which has an [Worker](../Workers/Workers.md) that is selected to complete task, and would engages the user for their response. With the user response, the TaskGraph will decide which next node to travel to.

While there are a lot of pre-built workers (and the ability to write custom workers too!), for this simple task, we will only be using two of the most basic workers. [MessageWorker](../Workers/MessageWorker.mdx) and [DefaultWorker](../Workers/DefaultWorker.mdx). Message Worker is the base worker in handling message responses and DefaultWorker is there in case MessageWorker is not deemed suitable.

Here is the simple structure for a [Config](../Config/intro.md) JSON file:

* `role (Required)`: The general "role" of the chatbot you want to create. For instance, "roleplay bot", "customer service assistant", "data analyst", "shopping assistant", etc.
* `user_objective (Required)`: The user's goal that the chatbot wants to achieve. Related to the user experience. Description in third person. For instance, "The customer service assistant helps users with customer service inquiries. It can provide information about products, services, and policies, as well as help users resolve issues and complete transactions."
* `builder_objective (Optional)`: The additional target you want the chatbot to achieve beyond the user's goal. Can contain hidden objectives or subtle objectives which is hidden from the user. Describe in third person. For instance, "The customer service assistant helps to request customer's contact information."
* `domain (Optional)`: The domain that you want to create the chatbot for. For instance, "robotics and automation", "Ecommerce", "Healthcare", etc.
* `intro (Optional)`: The introduction of the above domain that you want to create the chatbot for. It should contain the information about the domain, the products, services, and policies, etc.
* `task_docs (Optional, List[Dict])`: The documents resources for the taskgraph generation to create the chatbot. Each item in the list should contain the following fields:
  * `source (Required)`: The source url that you want the chatbot to refer to
  * `desc (Optional)` : Short description of the source and how it is used
  * `num (Optional)`: The number of websites that you want the chatbot to refer to for the source, defaults to one (only the url page)
* `rag_docs (Optional, List[Dict])`: If you want to use RAGWorker, then here indicates the documents for the RAG component of chatbot when running chatbot. Each item in the list should contain the following fields:
  * `source (Required)`: The source url that you want the chatbot to refer to
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

Now, lets see it with the Roleplay Bot example.

```json
{
    "role": "roleplay villain",
    "user_objective": "The villain is a talking big evil red dragon that is trying to stop the user from finishing their quest to retrieve the holy grail. It is deadly and have killed many knights and heros in the past.",
    "builder_objective": "",
    "domain": "action roleplay",
    "intro": "",
    "docs": [],
    "tasks": [],
    "workers": [
        {"id": "9244468a-5b0a-4bd2-99aa-650f0efeb0ac", "name": "MessageWorker", "path": "message_worker.py"}
    ]
}
```

With our Config in place, the vast majority of work is surprisingly already done! The rest is simply bringing the bot to life.

## Generating a TaskGraph

Now that we have a Config file, generating the graph is the easy part. All you need to do is run the following command:

`python create.py --config ./examples/roleplay.json --output-dir ./examples/roleplay`

It will first enter into a *task planning* interactive panel where you can see the generated tasks this bot will handle and the following steps to complete the specific tasks. You can also modify the tasks and steps as needed. Once you are satisfied with result, you could press `s` to save the *task planning* file then it will further generate the final *TaskGraph* file.

TaskGraph provides the graph that the bot will traverse through during the conversation. It provides a guideline for the conversation to make it more controllable and reliable. The details can be viewed at [here](../Taskgraph/Generation.md).

## Running the Bot

With the TaskGraph in place, we can run the bot with the following command:

`python run.py --input-dir ./examples/roleplay`

It will initialize the service (e.g. NLU apis) you need to start the bot and you can start interacting with it!

---

## Sample Conversation

Here is a sample conversation of the roleplay bot:

```
> Bot: Greetings, foolish adventurer! I am the mighty red dragon, guardian of the secrets you seek. Turn back now, lest you join the countless knights who have perished before you. What brings you to my fiery domain in pursuit of the elusive holy grail?

> User: I am questing for the holy grail for the hand of the princess!

> Bot: Greetings, intrepid seeker! How may I be of service to you today? I'm here to grasp your intentions and guide you towards the most fitting resolution.

> User: I want the grail.

> Bot: Ah, you dare to claim the grail, do you? Know this, brave soul: many have tried and met their doom at my claws. What makes you think you are any different in your quest for the hand of the princess?

> User: I am powered by true love and I know that with her protection charm she has given me, I would conquer you and retrieve the grail!   

> Bot: Ah, you dare to claim the grail, do you? Know this, brave soul: many have tried and met their doom at my claws. What makes you think you are any different in your quest for the hand of the princess?

> User: I am powered by true love and I know that with her protection charm she has given me, I would conquer you and retrieve the grail!   

> Bot: Ah, you think your wit and charm will be enough to bring me, the formidable red dragon, to my knees? Many have tried to outsmart and sway me with their clever words, only to meet their end in my fiery domain. What makes you so confident that your quest for the holy grail and the princess's hand will lead to a different fate?
```
