from dataclasses import dataclass
from langgraph.runtime import Runtime
from friday.graph.query.contexts.llm_context import LLMContext
from friday.graph.query.contexts.user_context import UserContext
from friday.graph.query.states.messages_state import MessagesState
from friday.loggers.logger import Logger


@dataclass
class QueryRewriterContext(LLMContext, UserContext):
    pass


async def query_rewriter(state: MessagesState, runtime: Runtime[QueryRewriterContext]):
    logger = Logger.get_logger("node.query_rewriter")
    logger.info("rewriting query")

    message = f"""You are in a conversation with a user, answering questions about {runtime.context.user_context}.
You are about to look up information in a Knowledge Base to answer the user's question.

This is the history of your conversation so far with the user:
{state['messages'][1:]}

And this is the user's current question:
{state['messages'][-1:]}

Respond only with a single, refined question that you will use to search the Knowledge Base.
It should be a VERY short specific question most likely to surface content. Focus on the question details.
Don't mention {runtime.context.user} unless it's a general question about {runtime.context.user}.
IMPORTANT: Respond ONLY with the knowledgebase query, nothing else."""
    logger.debug("message: %s", lambda: message)

    response = await runtime.context.llm.invoke(
        [{"role": "system", "content": message}]
    )
    logger.debug("response: %s", lambda: response)

    return {
        "messages": [
            *state["messages"][-1:],
            {
                "role": "user",
                "content": response.output_text,
            },
        ]
    }
