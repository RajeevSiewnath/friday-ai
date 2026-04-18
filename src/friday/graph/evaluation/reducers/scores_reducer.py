from copy import deepcopy
from typing import Any


def scores_reducer(left: dict[str, list[Any]], right: dict[str, list[Any]]):
    state = deepcopy(left)
    for key, values in right:
        if key not in left:
            left[key] = []
        left[key].extend(values)
    return state
