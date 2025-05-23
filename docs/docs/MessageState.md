# MessageState

`MessageState` is a data structure that represents the current snapshot of the conversation, which is developed on top of [LangGraph State](https://langchain-ai.github.io/langgraph/concepts/low_level/#state). It is the input parameter passed from the orchestrator to workers.

Here is the implementation of a sample `MessageState`:

```py
class MessageState(TypedDict):
    # system configuration
    sys_instruct: str
    # input message
    user_message: ConvoMessage
    orchestrator_message: OrchestratorMessage
    # message flow between different nodes
    message_flow: Annotated[str, "message flow between different nodes"]
    # final response
    response: str
    # task-related params
    status: StatusEnum
    slots: list[Slot]
```

- `sys_instruct`: The system-level instructions for the orchestrator.
- `user_message`: A `ConvoMessage` object containing the user's query and chat history.
- `orchestrator_message`: An `OrchestratorMessage` object contains value and attributes of a node in [TaskGraph](./Taskgraph/Generation.md).
- `message_flow`: Annotated[str, "message flow between different nodes"]
- `response`: The final response for the user after execution.
- `status`: A `StatusEnum` object indicating whether the task has been completed
- `slots`: A list of `Slot` objects to collect information during the conversation.

For the list of `Slot` objects, they should be defined as:

```py
class Slot(BaseModel):
    name: str
    type: str
    value: str
    description: str
    prompt: str
```
