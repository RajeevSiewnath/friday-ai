from dataclasses import dataclass
from friday.core.vector_db import VectorDB


@dataclass
class VectorDBContext:
    vector_db: VectorDB
