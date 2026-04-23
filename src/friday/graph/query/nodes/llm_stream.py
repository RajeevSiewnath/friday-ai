from langgraph.runtime import Runtime
from langgraph.config import get_stream_writer
from friday.graph.query.states.messages_state import MessagesState
from friday.graph.query.contexts.llm_context import LLMContext
from friday.loggers.logger import Logger


async def llm_stream(state: MessagesState, runtime: Runtime[LLMContext]):
    logger = Logger.get_logger("node.llm_stream")
    logger.info("streaming llm")
    logger.debug("messages: %s", lambda: state["messages"])

    writer = get_stream_writer()
    events = []
    async for event in runtime.context.llm.stream(state["messages"]):
        logger.debug("write: %s", lambda: event)
        writer({"messages": [event]})
        events.append(event)

    logger.debug("events: %s", lambda: events)
    return {"messages": events}
