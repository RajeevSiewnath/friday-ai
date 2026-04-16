from typing import Annotated, TypedDict
from friday.reducers.stream_reducer import stream_reducer


class MessagesState(TypedDict):
    messages: Annotated[list[dict], stream_reducer]
