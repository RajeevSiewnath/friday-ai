from typing import Annotated, TypedDict
from friday.core.vector_db import VectorQueryOutput
from friday.graph.query.reducers.rag_reducer import rag_reducer


class RagState(TypedDict):
    rag_data: Annotated[dict[str, list[VectorQueryOutput]], rag_reducer]
