from typing import Protocol
from friday.query_nodes.states.messages_state import MessagesState
from friday.query_nodes.states.system_prompt_state import SystemPromptState
from friday.reducers.system_prompt_reducer import SystemPromptReducerResetAction


class UpdatePromptContextHistoryState(MessagesState, SystemPromptState, Protocol):
    pass


def update_prompt_context_history(state: UpdatePromptContextHistoryState):
    system_prompt_id = next(
        [message["id"] for message in state["messages"] if message["type"] == "system"],
        None,
    )
    return {
        "messages": {
            "id": system_prompt_id,
            "content": SystemPromptReducerResetAction(state["system_prompt"]),
        }
    }
