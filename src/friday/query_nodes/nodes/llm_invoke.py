from langgraph.runtime import Runtime
from friday.query_nodes.states.messages_state import MessagesState
from friday.query_nodes.contexts.llm_context import QueryContext


async def llm_invoke(state: MessagesState, runtime: Runtime[QueryContext]):
    response = await runtime.context.llm.invoke(state["messages"])
    return {"messages": [m.model_dump() for m in response.output]}
