import asyncio
import io
import os
import gradio as gr
from dataclasses import dataclass
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from colorama import init
from PIL import Image
from friday.core.graph_invoker import GraphInvoker
from friday.core.llm import LLM
from friday.core.mcp_tool_box import MCPToolBox
from friday.core.tool_shed import ToolShed
from friday.core.vector_db import VectorDB
from friday.graph.query.contexts.llm_context import LLMContext
from friday.graph.query.contexts.vector_db_context import VectorDBContext
from friday.graph.query.edge_checks.is_function_call import is_function_call
from friday.graph.query.nodes.capabilities_injector import capabilities_injector
from friday.graph.query.nodes.mcp_server_open import mcp_server_open
from friday.graph.query.nodes.mcp_server_close import mcp_server_close
from friday.graph.query.nodes.llm_stream import llm_stream
from friday.graph.query.states.messages_state import MessagesState, messages_filter
from friday.graph.query.states.rag_state import RagState
from friday.graph.query.subs.tool_call_graph.tool_call_graph import tool_call_graph
from friday.tools.send_push_notification import send_push_notification

# from friday.graph.query.nodes.execute_tool import execute_tool
# from friday.debuggers.debug_chat import debug_chat


init(autoreset=True)

load_dotenv()


class State(MessagesState, RagState):
    pass


@dataclass
class Context(LLMContext, VectorDBContext):
    pass


nodes_image: Image = None


def chatbot_graph_invoker(context: Context) -> GraphInvoker:
    global nodes_image

    graph = StateGraph(State, context_schema=Context)
    graph.add_node("mcp_open", mcp_server_open)
    graph.add_node("capabilities_injector", capabilities_injector)
    graph.add_node("llm_stream", llm_stream)
    # graph.add_node("execute_tool", execute_tool)
    graph.add_node("execute_tool", tool_call_graph(context.llm).compile())
    graph.add_node("mcp_close", mcp_server_close)

    graph.add_edge(START, "mcp_open")
    graph.add_edge("mcp_open", "capabilities_injector")
    graph.add_edge("capabilities_injector", "llm_stream")
    graph.add_conditional_edges(
        "llm_stream",
        is_function_call("execute_tool", "mcp_close"),
        {edge: edge for edge in ["execute_tool", "mcp_close"]},
    )
    graph.add_edge("execute_tool", "llm_stream")
    graph.add_edge("mcp_close", END)

    compiled_graph = graph.compile()
    # if nodes_image is None:
    png_bytes = compiled_graph.get_graph(xray=True).draw_mermaid_png()
    nodes_image = Image.open(io.BytesIO(png_bytes))
    return GraphInvoker(compiled_graph, context=context)


async def invoke(state: State, context: Context):
    async for new_state in chatbot_graph_invoker(context).stream(state):
        # debug_chat(new_state["messages"])
        yield "", new_state, messages_filter(new_state["messages"]), None, nodes_image


def submit(context: Context):
    async def submit_message(message: str, state: State):
        state["messages"].append({"role": "user", "content": message})
        async for value in invoke(state, context):
            yield value

    return submit_message


async def run(context: Context):
    with gr.Blocks() as app:
        state: State = {
            "messages": [
                {
                    "role": "system",
                    "content": "You're the personal career agent of Rajeev Siewnath. Help the user with any career-related queries they have about Rajeev Siewnath.",
                },
                {"role": "assistant", "content": "Hi, how can I help you?"},
            ],
            "rag_data": {},
            "system_prompt": "",
        }
        gr_state = gr.State(state)
        with gr.Row():
            gr.Markdown("# Friday - AI")
        with gr.Row():
            with gr.Column():
                gr_chatbot = gr.Chatbot(messages_filter(gr_state.value["messages"]))
                gr_textfield = gr.Textbox(
                    autofocus=True,
                    placeholder="Chat with the bot!",
                    interactive=True,
                )
            with gr.Column():
                gr_nodes_image = gr.Image()
                gr_plot = gr.Plot(None)

        gr_textfield.submit(
            fn=submit(context),
            inputs=[gr_textfield, gr_state],
            outputs=[gr_textfield, gr_state, gr_chatbot, gr_plot, gr_nodes_image],
        )

    os.system("cls" if os.name == "nt" else "clear")
    app.launch()


async def setup() -> Context:
    mcp_tool_box = MCPToolBox({"command": "uvx", "args": ["mcp-server-fetch"]})
    tool_shed = ToolShed(send_push_notification, mcp_tool_box=mcp_tool_box)
    await tool_shed.load_mcp_tool_box_isolated()
    llm = LLM(tool_shed=tool_shed)
    vector_db = VectorDB()
    context: Context = Context(llm=llm, vector_db=vector_db)
    return context


async def main():
    context = await setup()
    await run(context)


if __name__ == "__main__":
    asyncio.run(main())
