def is_function_call(true: str, false: str):
    def check(state):
        return (
            true
            if state["messages"] and state["messages"][-1]["type"] == "function_call"
            else false
        )

    return check
