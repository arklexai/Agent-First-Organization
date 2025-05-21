from arklex.env.tools.tools import register_tool

# state in -> state out
#   no preprocessing
#   no postprocessing
"""
@register_tool("Repeats the user query back at them")
def echo(state: MessageState) -> MessageState:
    state.response = state.user_message.message
    return state
"""

# state in -> value out
#   no preprocessing
#   value post-processing
"""
@register_tool(
    "Repeats the user query back but in all caps",
    [{
        "name": "msg",
        "type": "string",
        "value": "",
        "description": "the most recent user message",
    }]
    )
def shoutEcho(msg: str) -> str:
    return msg.upper()
"""


# value in value out
#   value preprocessing - slots
#   value post-processing
@register_tool(
    "Calculates and return the function output of mathematical query.",
    [
        {
            "name": "expression",
            "type": "string",
            "description": "valid math expression extracted from the user message expressed with only numerical digits and these special characters ['(', ')', '+', '-', '*', '/', '%', '^']",
            "prompt": "Could you please provide the mathematical expression?",
            "required": True,
        }
    ],
    [
        {
            "name": "result",
            "type": "int",
            "value": "",
            "description": "result of evaluated mathematical expression",
        }
    ],
)
def calculator(expression: str) -> int:
    py_expression: str = expression.replace("^", "**")
    return eval(py_expression)
