from dataclasses import dataclass
from friday.core.llm import LLM


@dataclass
class LLMContext:
    llm: LLM
