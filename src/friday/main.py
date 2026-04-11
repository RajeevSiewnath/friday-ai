import asyncio
import json
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from friday.core.llm import LLM
from friday.core.graph_invoker import GraphInvoker
from friday.core.tool_shed import ToolShed
from friday.debuggers.chat_debuggers import debug_chat
from friday.reducers.stream_reducer import stream_reducer
from colorama import init
from langgraph.config import get_stream_writer
from langgraph.graph import END, StateGraph

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


class State(TypedDict):
    messages: Annotated[list[dict], stream_reducer]


tool_shed = ToolShed(send_contact_request)
llm = LLM(
    # api_key="ollama",
    # model="llama3.2",
    # base_url="http://localhost:11434/v1/",
    tool_shed=tool_shed,
)


async def chat_node_stream(state: State):
    writer = get_stream_writer()
    events = []
    async for event in llm.stream(state["messages"]):
        writer({"messages": [event]})
        events.append(event)

    return {"messages": events}


async def chat_node(state: State):
    response = await llm.invoke(state["messages"])
    return {"messages": [m.model_dump() for m in response.output]}


async def execute_tool(state: State):
    tool_call = state["messages"][-1]
    result = llm.tool_shed.call(tool_call["name"], tool_call["arguments"])
    return {
        "messages": [
            {
                "type": "function_call_output",
                "call_id": tool_call["call_id"],
                "output": str(result),
            }
        ]
    }


def build_graph() -> StateGraph:
    builder = StateGraph(State)
    builder.add_node("chat", chat_node)
    builder.add_node("tool", execute_tool)
    builder.set_entry_point("chat")
    builder.add_edge("tool", "chat")
    builder.add_conditional_edges(
        "chat",
        lambda state: (
            "tool" if state["messages"][-1]["type"] == "function_call" else END
        ),
    )
    return builder.compile()


async def main():
    graph = build_graph()
    graph_invoker = GraphInvoker(graph)

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
