from dataclasses import dataclass
from friday.core.llm import LLM
from friday.core.prompt_context import PromptContext
from friday.core.vector_db import VectorDB


@dataclass
class QueryContext:
    llm: LLM
    prompt_context: PromptContext
    vector_db: VectorDB
