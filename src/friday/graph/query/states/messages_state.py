from typing import Annotated, TypedDict
from friday.graph.query.reducers.stream_reducer import stream_reducer


class MessagesState(TypedDict):
    messages: Annotated[list[dict], stream_reducer]
