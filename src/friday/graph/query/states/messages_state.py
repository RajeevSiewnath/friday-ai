from typing import Annotated, TypedDict
from friday.graph.query.reducers.stream_reducer import stream_reducer
from friday.utils.print_content import print_content


class MessagesState(TypedDict):
    messages: Annotated[list[dict], stream_reducer]


def messages_filter(messages: list[dict]):
    return [
        {
            **m,
            "content": print_content(m["content"]),
        }
        for m in messages
        if "role" in m and "content" in m
    ]
