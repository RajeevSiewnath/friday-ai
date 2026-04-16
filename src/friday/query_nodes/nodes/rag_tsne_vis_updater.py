from typing import Protocol
from langgraph.runtime import Runtime
from friday.query_nodes.contexts.llm_context import LLMContext
from friday.query_nodes.states.messages_state import MessagesState
from friday.query_nodes.states.rag_state import RagState
from friday.visualizations.vector_db_tsne_visualization import VectorDBTSNEVisualization


class RagTNSEVisUpdatedState(MessagesState, RagState, Protocol):
    pass


def rag_tsne_vis_updater_factory(
    state_key: str, vector_db_tsne_vis_updater: VectorDBTSNEVisualization
):
    async def rag_tsne_vis_updater(
        state: RagTNSEVisUpdatedState, runtime: Runtime[LLMContext]
    ):
        vector_db_tsne_vis_updater.highlight_ids = [
            context.id for context in state["rag_data"][state_key]
        ]
        vector_db_tsne_vis_updater.question = (
            state["messages"][-1]["content"],
            await runtime.context.llm.embedding(state["messages"][-1]["content"]),
        )
        return {}

    return rag_tsne_vis_updater
