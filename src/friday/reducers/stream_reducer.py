from copy import deepcopy
import uuid


class StreamReducerReplaceAction:
    def __init__(self, content: str):
        self.content = content


def stream_reducer(left: list[dict], right: list[dict]):
    content = deepcopy(left)

    def get_message_by_id(id: str):
        return next(
            (i for i in content if id is not None and i.get("id", None) == id), None
        )

    for entry in right:
        message_in_left = get_message_by_id(entry.get("id", None))
        if message_in_left:
            if "content" in message_in_left and "content" in entry:
                left_content = message_in_left["content"]
                right_content = entry["content"]
                message_in_left.update(
                    {k: v for k, v in entry.items() if k != "content"}
                )
                if isinstance(right_content, StreamReducerReplaceAction):
                    message_in_left["content"] = right_content.content
                else:
                    message_in_left["content"] = f"{left_content}{right_content}"
            else:
                message_in_left.update(
                    {k: v for k, v in entry.items() if k != "content"}
                )
        else:
            copy = deepcopy(entry)
            if "id" not in copy:
                copy["id"] = str(uuid.uuid4())
            if isinstance(copy["content"], StreamReducerReplaceAction):
                copy["content"] = copy["content"].content
            content.append(copy)

    return content
