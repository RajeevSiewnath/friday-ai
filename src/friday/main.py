import asyncio
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from colorama import init
from friday.core.llm import LLM
from friday.core.graph_invoker import GraphInvoker
from friday.core.prompt_context import PromptContext
from friday.core.vector_db import VectorDB
from friday.debuggers.chat_debuggers import debug_chat
from friday.query_nodes.execute_tool import execute_tool
from friday.query_nodes.llm_invoke import llm_invoke
from friday.query_nodes.states.messages_state import MessagesState
from friday.query_nodes.contexts.llm_context import QueryContext
from friday.query_nodes.states.rag_state import RagState
from friday.query_nodes.states.system_prompt_state import SystemPromptState

init(autoreset=True)

load_dotenv()


def send_contact_request(message: str) -> bool:
    """
    Send a message to Rajeev Siewnath.

    Args:
        message: The message to send

    Returns:
        Whether the message was sent successfully
    """
    # print("Sending:", message)
    return True


llm = LLM(
    # api_key="ollama",
    # model="llama3.2",
    # base_url="http://localhost:11434/v1/",
)
llm.tool_shed.add(send_contact_request)
vector_db = VectorDB()

class State(MessagesState, RagState, SystemPromptState):
    pass


def build_graph() -> StateGraph:
    builder = StateGraph(State, context_schema=QueryContext)
    builder.add_node("llm_invoke", llm_invoke)
    builder.add_node("execute_tool", execute_tool)
    builder.set_entry_point("llm_invoke")
    builder.add_edge("execute_tool", "llm_invoke")
    builder.add_conditional_edges(
        "llm_invoke",
        lambda state: (
            "execute_tool" if state["messages"][-1]["type"] == "function_call" else END
        ),
    )
    return builder.compile()


async def main():
    graph = build_graph()
    graph_invoker = GraphInvoker(graph, context=query_context)

    state: State = {
        "messages": [
            {
                "role": "system",
                "content": "you are a helpful assistant. greet the user. you have the the capability to send messages to rajeev",
            },
            {"role": "user", "content": "send a message to rajeev: hey bro"},
        ]
    }

    while True:
        # async for s in graph_invoker.stream(state):
        #     debug_chat(s["messages"])
        #     news = s
        # state = news
        # newq = input(">")
        # state["messages"].append({"role": "user", "content": newq})
        news = await graph_invoker.invoke(state)
        debug_chat(news["messages"])
        state = news
        newq = input(">")
        state["messages"].append({"role": "user", "content": newq})


if __name__ == "__main__":
    asyncio.run(main())
