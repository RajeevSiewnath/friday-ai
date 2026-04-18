from friday.graph.query.states.rag_state import RagState
from friday.graph.query.states.system_prompt_state import SystemPromptState


class RagContextInjectorState(SystemPromptState, RagState):
    pass


def rag_context_injector_factory(state_key: str, context_label: str = "Context"):
    def rag_context_injector(state: RagContextInjectorState):
        return {
            "system_prompt": f"{context_label}:\n"
            + "\n".join(
                query_result.document for query_result in state["rag_data"][state_key]
            )
        }

    return rag_context_injector
