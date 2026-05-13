from copy import deepcopy


class StreamReducerReplaceAction:
    def __init__(self, content: dict):
        self.content = content


class StreamReducerClearAction:
    def __init__(self, content: dict):
        self.content = content


def stream_reducer(
    left: list[dict],
    right: list[dict | StreamReducerReplaceAction | StreamReducerClearAction],
):
    left_entries = deepcopy(left)

    def get_message_by_id(id: str):
        return next(
            (i for i in left_entries if id is not None and i.get("id", None) == id),
            None,
        )

    def get_system_message():
        return next((i for i in left_entries if i.get("role") == "system"), None)

    for entry in right:
        right_entry: dict = None
        replace = False
        if isinstance(entry, StreamReducerReplaceAction):
            right_entry = deepcopy(entry.content)
            replace = True
        elif isinstance(entry, StreamReducerClearAction):
            right_entry = deepcopy(entry.content)
            left_entries = []
        else:
            right_entry = deepcopy(entry)

        if right_entry.get("role") == "system":
            left_entry = get_system_message()
        else:
            left_entry = get_message_by_id(right_entry.get("id"))

        if left_entry:
            left_entry.update({k: v for k, v in right_entry.items() if k != "content"})
            if "content" in left_entry and "content" in right_entry:
                left_entry["content"] = (
                    right_entry["content"]
                    if replace
                    else f"{left_entry["content"]}{right_entry["content"]}"
                )
        else:
            left_entries.append(right_entry)

    return left_entries
