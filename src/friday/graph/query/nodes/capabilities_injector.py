from langgraph.runtime import Runtime
from friday.graph.query.contexts.llm_context import LLMContext
from friday.graph.query.states.system_prompt_state import SystemPromptState


def capabilities_injector(state: SystemPromptState, runtime: Runtime[LLMContext]):
    if len(runtime.context.llm.tool_shed.tools) > 0:
        return {
            "system_prompt": "Capabilities:\n"
            + "\n".join(
                f"- {tool.name}: {tool.definition["description"]}"
                for tool in runtime.context.llm.tool_shed.tools
            )
        }
    else:
        return {}
