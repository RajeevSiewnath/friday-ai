from dataclasses import dataclass
from langgraph.runtime import Runtime
from friday.graph.query.contexts.llm_context import LLMContext
from friday.graph.query.contexts.vector_db_context import VectorDBContext
from friday.graph.query.states.messages_state import MessagesState
from friday.graph.query.states.rag_state import RagState


class RagFromQueryState(RagState, MessagesState):
    pass


@dataclass
class RagFromQueryContext(LLMContext, VectorDBContext):
    pass


def rag_from_query_factory(collection_key: str, state_key: str | None = None):
    async def rag_from_query(
        state: RagFromQueryState, runtime: Runtime[RagFromQueryContext]
    ):
        embedding = await runtime.context.llm.embedding(
            state["messages"][-1]["content"]
        )
        query_result = runtime.context.vector_db[collection_key].query(
            embedding=embedding
        )
        key = state_key if state_key is not None else collection_key
        return {"rag_data": {key: query_result}}

    return rag_from_query
