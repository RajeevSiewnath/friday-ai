from dataclasses import dataclass
from langgraph.runtime import Runtime
from friday.graph.query.contexts.llm_context import LLMContext
from friday.graph.query.contexts.vector_db_context import VectorDBContext
from friday.graph.query.states.messages_state import MessagesState
from friday.graph.query.states.rag_state import RagState
from friday.loggers.logger import Logger


class RagFromQueryState(RagState, MessagesState):
    pass


@dataclass
class RagFromQueryContext(LLMContext, VectorDBContext):
    pass


def rag_from_query_factory(collection_key: str, state_key: str | None = None):
    async def rag_from_query(
        state: RagFromQueryState, runtime: Runtime[RagFromQueryContext]
    ):
        logger = Logger.get_logger("node.rag_from_query")
        logger.debug("generating rag context from query")
        logger.trace("query: %s", lambda: state["messages"][-1]["content"])

        embedding = await runtime.context.llm.embedding(
            state["messages"][-1]["content"]
        )

        query_result = runtime.context.vector_db[collection_key].query(
            embedding=embedding
        )
        logger.trace("query result: %s", lambda: query_result)

        key = state_key if state_key is not None else collection_key
        return {"rag_data": {key: query_result}}

    return rag_from_query
