from copy import deepcopy


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
                message_in_left["content"] = f"{left_content}{right_content}"
            else:
                message_in_left.update(
                    {k: v for k, v in entry.items() if k != "content"}
                )
        else:
            content.append(deepcopy(entry))

    return content
