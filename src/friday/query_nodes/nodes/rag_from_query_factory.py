from langgraph.runtime import Runtime
from friday.query_nodes.contexts.llm_context import LLMContext
from friday.query_nodes.contexts.vector_db_context import VectorDBContext
from friday.query_nodes.states.messages_state import MessagesState
from friday.query_nodes.states.rag_state import RagState


class RagFromQueryState(RagState, MessagesState):
    pass


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
