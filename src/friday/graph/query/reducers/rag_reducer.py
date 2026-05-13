from copy import deepcopy

from friday.core.vector_db import VectorQueryOutput


class RagReducerReplaceAction:
    def __init__(self, *content: VectorQueryOutput):
        self.content: list[VectorQueryOutput] = list(content)


def rag_reducer(
    left: dict[str, list[VectorQueryOutput]],
    right: dict[str, list[VectorQueryOutput | RagReducerReplaceAction]],
):
    output = deepcopy(left)
    if right is not None:
        for key, vector_data in right.items():
            if key not in output:
                output[key] = []
            if isinstance(vector_data, RagReducerReplaceAction):
                output[key] = vector_data.content
            else:
                output[key].extend(vector_data)
    return output
