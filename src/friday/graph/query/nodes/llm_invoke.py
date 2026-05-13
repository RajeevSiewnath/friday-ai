from langgraph.runtime import Runtime
from friday.graph.query.states.messages_state import MessagesState
from friday.graph.query.contexts.llm_context import LLMContext
from friday.loggers.logger import Logger


async def llm_invoke(state: MessagesState, runtime: Runtime[LLMContext]):
    logger = Logger.get_logger("node.llm_invoke")
    logger.debug("invoking llm")
    logger.trace("messages: %s", lambda: state["messages"])

    response = await runtime.context.llm.invoke(state["messages"])
    logger.trace("response: %s", lambda: response)

    return {"messages": [m.model_dump() for m in response.output]}
