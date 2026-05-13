from langgraph.runtime import Runtime
from friday.graph.query.contexts.llm_context import LLMContext
from friday.graph.query.states.messages_state import MessagesState
from friday.graph.query.states.rag_state import RagState
from friday.loggers.logger import Logger
from friday.visualizations.vector_db_tsne_visualization import VectorDBTSNEVisualization


class RagTNSEVisUpdatedState(MessagesState, RagState):
    pass


def rag_tsne_vis_updater_factory(
    vector_db_tsne_vis_updater: VectorDBTSNEVisualization, state_key: str = None
):
    async def rag_tsne_vis_updater(
        state: RagTNSEVisUpdatedState, runtime: Runtime[LLMContext]
    ):
        logger = Logger.get_logger("node.rag_tsne_vis_updater")
        logger.debug("updating t-SNE visualization for rag context")
        logger.trace("query: %s", lambda: state["messages"][-1]["content"])

        logger.trace(
            "highlights: %s",
            lambda: [
                context.id
                for context in state["rag_data"][
                    state_key or vector_db_tsne_vis_updater.collection.name
                ]
            ],
        )
        vector_db_tsne_vis_updater.highlight_ids = [
            context.id
            for context in state["rag_data"][
                state_key or vector_db_tsne_vis_updater.collection.name
            ]
        ]

        vector_db_tsne_vis_updater.question = (
            state["messages"][-1]["content"],
            await runtime.context.llm.embedding(state["messages"][-1]["content"]),
        )
        return {}

    return rag_tsne_vis_updater
