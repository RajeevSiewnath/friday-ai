from langgraph.runtime import Runtime
from friday.query_nodes.states.messages_state import MessagesState
from friday.query_nodes.contexts.llm_context import LLMContext


async def llm_invoke(state: MessagesState, runtime: Runtime[LLMContext]):
    response = await runtime.context.llm.invoke(state["messages"])
    return {"messages": [m.model_dump() for m in response.output]}
