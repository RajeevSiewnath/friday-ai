class SystemPromptReducerResetAction:
    def __init__(self, content: str):
        self.content = content


def system_prompt_reducer(left: str, right: str | SystemPromptReducerResetAction):
    if isinstance(right, SystemPromptReducerResetAction):
        return right.content
    else:
        return left + "\n\n" + right
