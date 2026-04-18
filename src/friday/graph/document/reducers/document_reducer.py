from copy import deepcopy

from friday.core.document import Document


class DocumentReducerClearAction:
    def __init__(self, content: list[Document]):
        self.content: list[Document] = content


def document_reducer(
    left: list[Document], right: DocumentReducerClearAction | list[Document]
):
    docs = deepcopy(left)

    if isinstance(right, DocumentReducerClearAction):
        return [*deepcopy(right.content)]
    else:
        for doc in right:
            current_index = next(
                [i for i, d in enumerate(docs) if d.id == doc.id], None
            )
            if current_index is not None:
                print(f"warning: {doc.id} already exists")
                docs[current_index] = deepcopy(doc)
            else:
                docs.append(doc)
        return [*deepcopy(left), *deepcopy(right)]
