from copy import deepcopy


class StreamReducerReplaceAction:
    def __init__(self, content: str):
        self.content = content


def stream_reducer(left: list[dict], right: list[dict]):
    content = deepcopy(left)

    def get_message_by_id(id: str):
        return next(
            (i for i in content if id is not None and i.get("id", None) == id), None
        )

    def get_system_message():
        return next(
            (i for i in content if i.get("role") == "system"),
            {"role": "system", "content": ""},
        )

    for entry in right:
        message_in_left = get_message_by_id(entry.get("id"))
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
        elif entry.get("role") == "system":
            message_in_left = get_system_message()
            right_content = entry["content"]
            message_in_left.update({k: v for k, v in entry.items() if k != "content"})
            if isinstance(right_content, StreamReducerReplaceAction):
                message_in_left["content"] = right_content.content
            else:
                message_in_left["content"] = (
                    f"{message_in_left["content"]}{right_content}"
                )
        else:
            copy = deepcopy(entry)
            if isinstance(copy.get("content"), StreamReducerReplaceAction):
                copy["content"] = copy["content"].content
            content.append(copy)

    return content
