import asyncio
import os
import random
import gradio as gr
from typing import Annotated, AsyncGenerator, Awaitable, List
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import add_messages
from langgraph.config import get_stream_writer
from typing_extensions import TypedDict
from friday.core.llm import LLM
from friday.debuggers.debug_chat import debug_chat
from friday.graph.query.reducers.stream_reducer import stream_reducer
from colorama import init

init(autoreset=True)

load_dotenv()


class ChatState(TypedDict):
    messages: Annotated[list[dict], add_messages]
    test: Annotated[list[str], lambda i, o: [*i, *o]]


def trace(node_name: str, print_state: bool = True):
    def decorator(fn: callable):
        def wrapper(state: dict):
            print(f"→ {node_name}{f": {state}" if print_state else ""}")
            result = fn(state)
            print(f"✓ {node_name}{f": {result}" if print_state else ""}")
            return result

        return wrapper

    return decorator


# model = init_chat_model(model="gpt-4.1-nano")


@trace("llm")
def llm_node(state: ChatState) -> ChatState:
    # response = await model.ainvoke(state["messages"])
    # return {"messages": [response]}
    return {
        "test": ["".join(random.choice("ABCDEFGHIJKLMNOPQRSTUVW") for _ in range(8))]
    }


graph = (
    StateGraph(ChatState)
    .add_node("llm", llm_node)
    .add_edge(START, "llm")
    .add_edge("llm", END)
    .compile()
)

graph_image = (
    None  # Image.open(io.BytesIO(graph.get_graph(xray=True).draw_mermaid_png()))
)


def invoke(text: str, input_state: ChatState):
    input_state["test"].append(text)
    print()
    print(f"→ START: {input_state}")
    output_state: ChatState = graph.invoke(input_state)
    print(f"✓ END: {output_state}")
    return "", output_state["test"], output_state


def generator():
    for i in range(10):
        yield i


class AState(TypedDict):
    numbers: Annotated[List[int], lambda i, o: [*i, *o]]


class PAState(AState):
    finished: False


@trace("up_node")
def up_node(state: AState) -> PAState:
    next_val = next(generator(), None)
    print(next_val)
    if next_val != None:
        return {"numbers": [next_val], "finished": False}
    else:
        return {"finished": True}


a_graph = (
    StateGraph(AState)
    .add_node("up", up_node)
    .add_edge(START, "up")
    .add_edge("up", END)
    .compile()
)


main_graph = (
    StateGraph(AState)
    .add_node("reup", a_graph)
    .add_edge(START, "reup")
    .add_conditional_edges(
        "reup",
        lambda state: END if "finished" in state and state["finished"] else "reup",
    )
    .compile()
)


def run_agraph(input_state):
    print(f"→ START: {input_state}")
    output_state = main_graph.invoke(input_state)
    print(f"✓ END: {output_state}")
    return output_state


def main():
    with gr.Blocks() as app:
        gr.Markdown("# 🤖 Friday")

        # Unique thread per session; history mirrors the chatbot display
        # config_state = gr.State(
        #     lambda: {"configurable": {"thread_id": str(uuid.uuid4())}}
        # )
        state = gr.State({"messages": [], "test": []})
        with gr.Row():
            with gr.Column():
                # chatbot = gr.Chatbot(height=500)
                chatbot = gr.TextArea()
                textbox = gr.Textbox(placeholder="Chat with Friday...", autofocus=True)
                textbox.submit(
                    invoke, inputs=[textbox, state], outputs=[textbox, chatbot, state]
                )
            with gr.Column():
                gr.Image(graph_image, interactive=False)
        # textbox.submit(
        #     chat,
        #     inputs=[textbox, history_state, config_state],
        #     outputs=[chatbot, textbox],
        # ).then(
        #     lambda h: h,  # sync history_state with whatever chatbot now shows
        #     inputs=[chatbot],
        #     outputs=[history_state],
        # )

    os.system("cls" if os.name == "nt" else "clear")
    app.launch()


class CState(TypedDict):
    messages: Annotated[list[dict], stream_reducer]


llm = LLM(api_key="ollama", model="llama3.2", base_url="http://localhost:11434/v1/")


async def chat_node(state: CState):
    writer = get_stream_writer()
    events = []
    # print(state["messages"][0])
    # writer((state["messages"][0], state["messages"]))
    events.append(state["messages"][0])
    async for event in llm.stream(state["messages"]):
        writer(event)
        events.append(event[0])

    return {"messages": events}


async def chat_node_buuu(state: CState):
    response = await llm.invoke_raw(state["messages"])
    return {"messages": [m.model_dump() for m in response.output]}


def build_graph() -> StateGraph:
    builder = StateGraph(CState)
    builder.add_node("chat", chat_node)
    builder.set_entry_point("chat")
    builder.add_edge("chat", END)
    return builder.compile()


async def invoke_graph_a(graph: CompiledStateGraph, state: CState) -> Awaitable[CState]:
    new_state = await graph.ainvoke(state)
    return new_state


async def invoke_graph(
    graph: CompiledStateGraph, state: CState
) -> AsyncGenerator[CState]:
    # print("── stream chunks ──────────────────────────────")
    # print(state)
    async for chunk in graph.astream(
        state, stream_mode=["custom", "updates", "values"], version="v2"
    ):
        # print(chunk)
        if chunk["type"] == "values":
            # ValuesStreamPart — full state snapshot after each step
            # print(f"State: {json.dumps(chunk['data'])}")
            pass
            # yield chunk["data"]
        elif chunk["type"] == "updates":
            # UpdatesStreamPart — only the changed keys from each node
            for node_name, state in chunk["data"].items():
                # print(f"Node `{node_name}` updated: {json.dumps(state)}")
                pass
            pass
        elif chunk["type"] == "messages":
            # MessagesStreamPart — (message_chunk, metadata) from LLM calls
            msg, metadata = chunk["data"]
            # print(msg.content, end="", flush=True)
            pass
        elif chunk["type"] == "custom":
            # CustomStreamPart — arbitrary data from get_stream_writer()
            # print(f"Custom: {json.dumps(chunk['data'])}")
            pass
            msg, full = chunk["data"]
            # print(full)
            yield {"messages": full}
    # print("── done ───────────────────────────────────────")


async def mmmain():
    graph = build_graph()

    state: CState = {
        "messages": [
            {
                "role": "system",
                "content": "you are a helpful assistant. greet the user",
            },
            {"role": "user", "content": ""},
        ]
    }

    while True:
        # news = await invoke_graph(graph, state)
        # debug_chat(news["messages"])

        async for s in invoke_graph(graph, state):
            # print(s["messages"])
            debug_chat(s["messages"])
            news = s

        # print("done")
        state = news
        newq = input(">")
        state["messages"].append({"role": "user", "content": newq})


if __name__ == "__main__":
    # main()
    # run_agraph({"numbers": [10]})
    # messages = [{"content": "hi", "role": "user"}]
    # llm = LLM(api_key="ollama", model="llama3.2", base_url="http://localhost:11434/v1/")
    # for event in llm.stream(messages):
    #     messages = stream_reducer(messages, [event])
    #     print(messages)
    asyncio.run(mmmain())
