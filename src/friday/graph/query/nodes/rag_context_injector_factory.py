from dataclasses import dataclass
from friday.graph.query.states.rag_state import RagState
from friday.graph.query.states.messages_state import MessagesState
from friday.loggers.logger import Logger


@dataclass
class RagContextInjectorState(MessagesState, RagState):
    pass


def rag_context_injector_factory(state_key: str, context_label: str = "Context"):
    def rag_context_injector(state: RagContextInjectorState):
        logger = Logger.get_logger("node.rag_context_injector")
        logger.info("injecting rag context into system prompt")
        logger.debug("rag data: %s", lambda: state["rag_data"][state_key])
        return {
            "messages": [
                {
                    "role": "system",
                    "content": f"\n\n{context_label}:\n"
                    + "\n".join(
                        query_result.document
                        for query_result in state["rag_data"][state_key]
                    ),
                }
            ]
        }

    return rag_context_injector
