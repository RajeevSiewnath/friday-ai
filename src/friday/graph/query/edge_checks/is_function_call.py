from typing import Literal


def is_function_call(true: str, false: str) -> Literal["execute_tool", "mcp_close"]:
    def check(state):
        return (
            true
            if state["messages"] and state["messages"][-1]["type"] == "function_call"
            else false
        )

    return check
