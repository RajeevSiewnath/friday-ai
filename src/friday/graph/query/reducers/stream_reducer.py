from copy import deepcopy


class StreamReducerReplaceAction:
    def __init__(self, content: str):
        self.content = content


def stream_reducer(left: list[dict], right: list[dict]):
    left_entries = deepcopy(left)

    def get_message_by_id(id: str):
        return next(
            (i for i in left_entries if id is not None and i.get("id", None) == id),
            None,
        )

    def get_system_message():
        return next((i for i in left_entries if i.get("role") == "system"), None)

    for entry in right:
        right_entry = deepcopy(entry)
        if right_entry.get("role") == "system":
            left_entry = get_system_message()
        else:
            left_entry = get_message_by_id(right_entry.get("id"))

        if left_entry:
            if "content" in left_entry and "content" in right_entry:
                left_content = left_entry["content"]
                right_content = right_entry["content"]
                left_entry.update(
                    {k: v for k, v in right_entry.items() if k != "content"}
                )
                if isinstance(right_content, StreamReducerReplaceAction):
                    left_entry["content"] = right_content.content
                else:
                    left_entry["content"] = f"{left_content}{right_content}"
            else:
                left_entry.update({k: v for k, v in entry.items() if k != "content"})
        else:
            if "content" in right_entry:
                if isinstance(right_entry["content"], StreamReducerReplaceAction):
                    right_entry["content"] = right_entry["content"].content
            left_entries.append(right_entry)

    return left_entries
