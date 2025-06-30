# Config

**Config** files are the simple and recommended way to build your [TaskGraph](../Taskgraph/intro.md). A standard JSON document, the [Generator](../Taskgraph/Generation) can create and breakdown the role into a series of tasks and steps which are then matched with the appropriate [Workers](../Workers/intro) and connected with the proper tasks to create a TaskGraph.

Here is the structure for a **Config** JSON file:

* `role (Required)`: The general "role" of the chatbot you want to create. For instance, "customer service assistant", "data analyst", "shopping assistant", etc.
* `user_objective (Required)`: The user's goal that the chatbot wants to achieve. Related to the user experience. Description in third person. For instance, "The customer service assistant helps users with customer service inquiries. It can provide information about products, services, and policies, as well as help users resolve issues and complete transactions."
* `builder_objective (Optional)`: The additional target you want the chatbot to achieve beyond the user's goal. Can contain hidden objectives or subtle objectives which is hidden from the user. Describe in third person. For instance, "The customer service assistant helps to request customer's contact information."
* `domain (Optional)`: The domain that you want to create the chatbot for. For instance, "robotics and automation", "Ecommerce", "Healthcare", etc.
* `intro (Optional)`: The introduction of the above domain that you want to create the chatbot for. It should contain the information about the domain, the products, services, and policies, etc.
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
* `settings (Optional, Dict)`: The extra configuration settings that you want to use for the chatbot. The dictionary should contain the following fields:
             * `hitl_proposal (Optional)` : Enable the HITL (human in the the loop) postprocess checking. When enabled, if the bot truly doesn't know the answer and question is relevant, it will ask the user if they want to connect to human assistant. Set to `True` or `False`
## Examples

#### [Customer Service Bot](../tutorials/customer-service.md)

```json title="customer_service_config.json"
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
    "tools": [],
    "settings": {}
}
```
