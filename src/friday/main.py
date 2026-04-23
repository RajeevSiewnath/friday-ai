import os
import gradio as gr
from dataclasses import dataclass
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from colorama import init
from friday.core.graph_invoker import GraphInvoker
from friday.core.llm import LLM
from friday.core.vector_db import VectorDB
from friday.graph.query.contexts.llm_context import LLMContext
from friday.graph.query.contexts.vector_db_context import VectorDBContext
from friday.graph.query.edge_checks.is_function_call import is_function_call
from friday.graph.query.nodes.execute_tool import execute_tool
from friday.graph.query.nodes.llm_stream import llm_stream
from friday.graph.query.states.messages_state import MessagesState, messages_filter
from friday.graph.query.states.rag_state import RagState
from friday.graph.query.states.system_prompt_state import SystemPromptState
from friday.tools.send_push_notification import send_push_notification
# from friday.debuggers.debug_chat import debug_chat


init(autoreset=True)

load_dotenv()


class State(MessagesState, RagState, SystemPromptState):
    pass


@dataclass
class Context(LLMContext, VectorDBContext):
    pass


llm = LLM()
llm.tool_shed.add(send_push_notification)
vector_db = VectorDB()


def chatbot_graph_invoker():
    graph = StateGraph(State, context_schema=Context)
    graph.add_node("llm_stream", llm_stream)
    graph.add_node("execute_tool", execute_tool)
    graph.set_entry_point("llm_stream")
    graph.add_edge("execute_tool", "llm_stream")
    graph.add_conditional_edges("llm_stream", is_function_call("execute_tool", END))
    compiled_graph = graph.compile()
    context: Context = Context(llm=llm, vector_db=vector_db)
    return GraphInvoker(compiled_graph, context=context)


async def submit(message: str, state: State):
    state["messages"].append({"role": "user", "content": message})
    async for new_state in chatbot_graph_invoker().stream(state):
        # debug_chat(new_state["messages"])
        yield "", new_state, messages_filter(new_state["messages"]), None


def main():
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
                chatbot = gr.Chatbot(messages_filter(gr_state.value["messages"]))
                textfield = gr.Textbox(
                    autofocus=True,
                    placeholder="Chat with the bot!",
                    interactive=True,
                )
            with gr.Column():
                plot = gr.Plot(None)

        textfield.submit(
            fn=submit,
            inputs=[textfield, gr_state],
            outputs=[textfield, gr_state, chatbot, plot],
        )

    os.system("cls" if os.name == "nt" else "clear")
    app.launch()


if __name__ == "__main__":
    main()
