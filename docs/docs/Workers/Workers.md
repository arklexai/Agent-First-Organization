# Workers

# Introduction

Workers serve as one of the fundamental building block of the **Agent First Organization** framework and responsible for the "execution" of the tasks and its subtasks. A structured component, *Workers* can be designed to do high level complicated tasks and can call other workers itself, greatly enhancing its ability to address. Workers takes in a [MessageState](MessageState.md) and returns a MessageState.

# Implementation

:::tip  Technical Details
Each Worker represent a node of GraphNodes or a  subgraph on the LangGraph.
:::

BaseWorker is the parent class of all workers is the BaseWorker and consists of a `description` attribute and `execute()` abstract method. Each node consists of a chain of relevants Workers by the orchestrator on execution which the on execution, processes the MessageState input into an MessageState output.

### Attributes

#### Description

`description` is a string that describes the task that the BaseWorker is meant to handle. This is used by the Generator to assign the right workers for the task when generating the TaskGraph.

Some examples:
    - "The worker that used to deliver the message to the user, either a question or provide some information." ([MessageWorker](./MessageWorker.mdx))
    - ""Answer the user's questions based on the company's internal documentations (unstructured text data), such as the policies, FAQs, and product information" ([RAGWorker](./RAGWorker.mdx))

#### Execute

`execute()` takes in [MessageState](MessageState.md) and returns an `invoke`'d LangChain StateGraph. This is crucial to connect various Workers (through LangChain subgraph behavior) and is called by the Orchestrator during runtime.

An example:

```py
def execute(self, msg_state: MessageState):
    graph = self.action_graph.compile()
    result = graph.invoke(msg_state)
    return result
```

### Code

```py
class BaseWorker(ABC):
    
    description = None

    def __str__(self):
        return f"{self.__class__.__name__}"

    def __repr__(self):
        return f"{self.__class__.__name__}"
    
    @abstractmethod
    def execute(self, msg_state: MessageState):
        pass
```
