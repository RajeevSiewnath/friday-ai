from copy import deepcopy

from friday.core.vector_db import VectorQueryOutput


class RagReducerReplaceAction:
    pass


def rag_reducer(
    left: dict[str, list[VectorQueryOutput]],
    right: dict[str, list[VectorQueryOutput | RagReducerReplaceAction]],
):
    output = deepcopy(left)
    for key, vector_data in right.items():
        if key not in output:
            output[key] = []
        if isinstance(vector_data, RagReducerReplaceAction):
            output[key] = []
        output[key].extend(vector_data)
    return output
