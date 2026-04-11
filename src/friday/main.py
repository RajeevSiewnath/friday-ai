import asyncio
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from friday.core.llm import LLM
from friday.core.graph_invoker import GraphInvoker
from friday.debuggers.chat_debuggers import debug_chat
from friday.reducers.stream_reducer import stream_reducer
from colorama import init
from langgraph.config import get_stream_writer
from langgraph.graph import END, StateGraph

init(autoreset=True)

load_dotenv()


class State(TypedDict):
    messages: Annotated[list[dict], stream_reducer]


llm = LLM(api_key="ollama", model="llama3.2", base_url="http://localhost:11434/v1/")


async def chat_node(state: State):
    writer = get_stream_writer()
    events = []
    async for event in llm.stream(state["messages"]):
        writer({"messages": [event]})
        events.append(event)

    return {"messages": events}


async def chat_node_no_stream(state: State):
    response = await llm.invoke(state["messages"])
    return {"messages": [m.model_dump() for m in response.output]}


def build_graph() -> StateGraph:
    builder = StateGraph(State)
    builder.add_node("chat", chat_node)
    builder.set_entry_point("chat")
    builder.add_edge("chat", END)
    return builder.compile()


async def main():
    graph = build_graph()
    graph_invoker = GraphInvoker(graph)

    state: State = {
        "messages": [
            {
                "role": "system",
                "content": "you are a helpful assistant. greet the user",
            },
            {"role": "user", "content": ""},
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
