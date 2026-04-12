from langgraph.runtime import Runtime
from langgraph.config import get_stream_writer
from friday.query_nodes.messages_state import MessagesState
from friday.query_nodes.query_context import QueryContext


async def llm_stream(state: MessagesState, runtime: Runtime[QueryContext]):
    writer = get_stream_writer()
    events = []
    async for event in runtime.context.llm.stream(state["messages"]):
        writer({"messages": [event]})
        events.append(event)

    return {"messages": events}
