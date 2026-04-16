from typing import Annotated, TypedDict
from friday.reducers.system_prompt_reducer import system_prompt_reducer


class SystemPromptState(TypedDict):
    system_prompt: Annotated[str, system_prompt_reducer]
