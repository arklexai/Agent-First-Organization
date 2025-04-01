# Tools

# Introduction
Alongside [Workers](./Workers), **Tools** are a fundamental building block of the **Agent First Organization** framework and responsible for the "execution" of the tasks and its subtasks. Tools are the unstructured counterpart to Workers, with a flexible input and output (contrary to Workers with a [MessageState](./MessageState.md) input and output). Powered through the SlotFilling mechanism, the framework handles the translation between MessageState and the specified parameter and return values, allowing smoother development of custom methods and integration of API. 

# Implementation
:::tip  Technical Details
Tools could be thought of as a Worker-wrapped method that has the framework translate values to and from MessageStates.
:::

Unlike Workers, there are no BaseWorker equivalent for Tools as it is designed to be unstructured. Instead, Tools could be easily created from any method through `@register-tool` decorator. To turn a method into a tool, just add a `@register-tool` decorator on the method declaration with 4 parameters: **description**, **slots**, **outputs**.

## Parameters
### Description
`description` is a string that describes what the Tools does and what tasks is meant to handle. This is used by the Generator to assign the right tool for the tool when generating the TaskGraph.

Example include:\
&emsp;"Adds two numbers and returns the sum"\
&emsp;"Searches the database for shows given descriptions"\
&emsp;"Add items to cart with line ids"

### Slots
Slots are the frameworks approach to extract values from the conversations. Slots are used in a tooling context to extract the parameters values from the conversation. Each parameter should have a corresponding slot. Through specifying the parameter's `name`, `type`, `description`, `prompt`, `required`, the framework would can extract the necessary information from the conversation. 

**`name`**\
A string representing the name of the parameter, this should be the parameter name.
> **Examples**: 'x', 'show_name', 'ids'

**`type`**: A string representing the type of data. 
> **Examples**: 'int', 'str', 'list'

**`enum`**: The candidate values for the slot. This is used to aid the slotfilling verifier to check if the extracted value is valid.
> **Examples**: 'int', 'str'

**`description`**: A string describing the parameter. This will be used to aid the extraction component of slotfilling. Adding examples often help the slotfilling.
> **Examples**\
> &emsp;"Name of the show, such as 'Hamilton'"\
> &emsp;"list of (item_id, quantity) tuples of Items to remove to the cart such as [('41552094527601', 5), ('41552094494833', 10)]."

**`prompt`**: A string representing the prompt if parameter required but not found
> **Examples**\
> &emsp;"Please provide the name of the show"

**`required`**: A boolean representing if the parameter is required. If required and a value could not be extracted, the prompt would be returned for the value

**`verified`**: A boolean representing whether the extracted parameter need verification from user. If need verification from user, then set as False, means haven't been verified. If not, set as True, means it has been verified. 

#### Example Slots
```py
[
    {
        "name": "cart_id",
        "type": "str",
        "description": "Cart ID to add items to, such as '2938501948327'",
        "prompt": "",
        "required": True,
        "verified": True
    },
    {
        "name": "item_ids",
        "type": "list",
        "items": "tuples"
        "description": "list of (item_id, quantity) tuples of Items to add to the cart such as [('41552094527601', 5), ('41552094494833', 10)].",
        "prompt": "",
        "required": True,
        "verified": True
    }
]
```

### Outputs
Outputs describes the expected return value of the method and aids the framework to contextualize the output of the tool for further Taskgraph decisions. Outputs are similar to slots except without `prompt` or `required` attributes.

> **Example**
> ```py 
> [{
>    "name": "user_id",
>    "type": "string",
>    "description": "The user id of the user. such as '13573257450893'.",
> }]
> ```

**Always Complete**
```py
lambda x: return True
```

**Error Messages**
```py
lambda x: return x not in ERROR_MSGS
```

**Not None**
```py
lambda x: return x is not None
```

## Examples
### Decorator
```py
@register_tool(
    "Add items to cart",
    [
        {
            "name": "cart_id",
            "type": "str",
            "description": "Cart ID to add items to, such as '2938501948327'",
            "prompt": "",
            "required": True,
            "verified": True
        },
        {
            "name": "item_ids",
            "type": "list",
            "items": "tuples"
            "description": "list of (item_id, quantity) tuples of Items to add to the cart such as [('41552094527601', 5), ('41552094494833', 10)].",
            "prompt": "",
            "required": True,
            "verified": True
        }
    ],
    [{
        "name": "cart",
        "type": "dict",
        "description": "The cart information after adding, such as {'id': 'sample_cart', 'items': {'41552094527601': 5, '41552094494833': 10\}\}.",
    }],
    lambda x: x is not None and x not in ERROR_MSGS
)
```

### Custom Tool
```py
import ast

@register_tool(
    "Calculates and return the function output of mathematical query.",
    [{
        "name": "expression",
        "type": "string",
        "description": "valid math expression extracted from the user message expressed with only numerical digits and these special characters ['(', ')', '+', '-', '*', '/', '%', '^'], like '21 * 2'",
        "prompt": "Could you please provide the mathematical expression?",
        "required": True
    }],
    [{
        "name": "result",
        "type": "float",
        "value": "",
        "description": "result of evaluated mathematical expression like 42",
    }],
    lambda x: isinstance(x, (int, float, complex)) and not isinstance(x, bool)
)
def calculator(expression):
    py_expression = expression.replace("^", "**")
    return ast.eval(py_expression)
```

### Convert Existing Method into a Tool
```py
from ast import eval as calculate

register_tool(
    "Calculates and return the function output of mathematical query.",
    [{
        "name": "expression",
        "type": "string",
        "description": "valid math expression extracted from the user message expressed with only numerical digits and these special characters ['(', ')', '+', '-', '*', '/', '%', '^'], like '21 * 2'",
        "prompt": "Could you please provide the mathematical expression?",
        "required": True
    }],
    [{
        "name": "result",
        "type": "float",
        "value": "",
        "description": "result of evaluated mathematical expression like 42",
    }],
    lambda x: isinstance(x, (int, float, complex)) and not isinstance(x, bool)
)(
    calculate
)
```