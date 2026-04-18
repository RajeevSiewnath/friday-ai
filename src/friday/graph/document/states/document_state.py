from typing import Annotated, TypedDict
from friday.core.document import Document
from friday.graph.document.reducers import document_reducer


class DocumentState(TypedDict):
    documents: Annotated[list[Document], document_reducer]
