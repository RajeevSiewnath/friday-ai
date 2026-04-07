import asyncio
import json
import os
import uuid
from typing import Annotated, List

import gradio as gr
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    messages_to_dict,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

load_dotenv()


class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


model = init_chat_model(model="gpt-4.1-nano")


async def llm_node(state: ChatState):
    response = await model.ainvoke(state["messages"])
    return {"messages": [response]}


memory = MemorySaver()
graph = (
    StateGraph(ChatState)
    .add_node("llm", llm_node)
    .add_edge(START, "llm")
    .add_edge("llm", END)
    .compile(checkpointer=memory)
)


def lc_to_api_format(messages: list[AnyMessage]):
    """
    Converts LangChain messages to the standard LLM dict format:
    [{"role": "user|assistant|system", "content": "..."}]
    """
    api_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "user"
            content = msg.content
        elif isinstance(msg, AIMessage):
            role = "assistant"
            content = msg.content
        elif isinstance(msg, SystemMessage):
            role = "system"
            content = msg.content
        else:
            # fallback if it's some custom AnyMessage subclass
            role = msg["role"]
            content = msg["content"][0]["text"]

        api_messages.append({"role": role, "content": content})

    return api_messages


async def chat(message: str, history: List[AnyMessage], config: dict):
    # Append user message optimistically
    history = history + [HumanMessage(content=message), AIMessage(content="")]
    yield lc_to_api_format(history), ""  # clear textbox immediately

    # Stream the assistant reply into the last history slot
    async for event in graph.astream(
        {"messages": [HumanMessage(content=message)]},
        config=config,
        stream_mode=["messages", "values"],
        version="v2",
    ):
        if event["type"] == "messages":
            history[-1].content += event["data"][0].content
            yield lc_to_api_format(history), gr.skip()
        if event["type"] == "values":
            print(event["data"])


def main():
    with gr.Blocks() as app:
        gr.Markdown("# 🤖 Friday")

        # Unique thread per session; history mirrors the chatbot display
        config_state = gr.State(
            lambda: {"configurable": {"thread_id": str(uuid.uuid4())}}
        )
        history_state = gr.State([])

        chatbot = gr.Chatbot(height=500)
        textbox = gr.Textbox(placeholder="Chat with Friday...", autofocus=True)

        textbox.submit(
            chat,
            inputs=[textbox, history_state, config_state],
            outputs=[chatbot, textbox],
        ).then(
            lambda h: h,  # sync history_state with whatever chatbot now shows
            inputs=[chatbot],
            outputs=[history_state],
        )

    os.system("cls" if os.name == "nt" else "clear")
    app.launch()


if __name__ == "__main__":
    main()
