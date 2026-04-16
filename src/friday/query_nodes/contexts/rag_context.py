from dataclasses import dataclass
from friday.core.vector_db import VectorQueryOutput


@dataclass
class RagContext:
    rag_data: dict[str, list[VectorQueryOutput]]
