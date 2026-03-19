import os
from agent_pipes.send_push_notification import SendPushNotification
from agent_pipelines.send_contact_request import SendContactRequest
from core.chat_loop import ChatLoop
from core.llm import LLM
from core.prompt_context import PromptContext
import gradio as gr
from core.vector_db import VectorDB
from core_pipes.invoke_chat_loop import InvokeChatLoop
from models.query_context import QueryContext
from pipelines.pipeline_factory import PipelineFactory
from query_pipes.capabilities_injector import CapabilitiesInjector
from query_pipes.query_writer import QueryWriter
from query_pipes.rag_context_injector import RagContextInjector
from query_pipes.rag_context_retriever import RagContextRetriever
from query_pipes.rag_tsne_vis_updater import RagTSNEVisUpdater
from query_pipes.update_prompt_context_history import UpdatePromptContextHistory
from visualizations.vector_db_tsne_visualization import VectorDBTSNEVisualization

llm = LLM()
prompt_context = PromptContext(
    user_context="""
You are a personal job agent for Rajeev Siewnath. 
You provide information about his curriculum vitae to the user, who is a person interested in Rajeev Siewnath's career.
If relevant, use the given context to answer any question."""
)
vector_db = VectorDB(llm=llm, name="cv-rajeev-siewnath")
chat_loop = ChatLoop(llm=llm)

vector_db_tsne_visualization = VectorDBTSNEVisualization(
    title="tSNE", collection=vector_db.collection
)

pipeline_factory = PipelineFactory(
    llm=llm,
    prompt_context=prompt_context,
    vector_db=vector_db,
)

query_pipeline = (
    pipeline_factory.make(QueryWriter())
    .pipe(
        RagContextRetriever(n_results=5),
        RagContextInjector(),
        RagTSNEVisUpdater(vector_db_tsne_visualization),
        CapabilitiesInjector(),
        UpdatePromptContextHistory(),
    )
    .pipe(InvokeChatLoop(chat_loop=chat_loop))
)

send_push_notification_pipeline = pipeline_factory.make(
    SendPushNotification(), pipeline_cls=SendContactRequest
)
llm.tool_shed.add(send_push_notification_pipeline)

vector_db_tsne_figure = vector_db_tsne_visualization.get()


def submit_message(question):
    if not chat_loop.is_looping and question:
        for done in query_pipeline.run(question):
            yield gr.Textbox(
                value="", placeholder="Working..." if not done else "Chat with the bot!"
            ), prompt_context.conversation, (
                vector_db_tsne_figure
                if not done
                else vector_db_tsne_visualization.get()
            )
    else:
        yield gr.Textbox(
            value="", placeholder="Chat with the bot!"
        ), prompt_context.conversation, vector_db_tsne_figure


def main():
    prompt_context.reset()
    prompt_context.push({"role": "assistant", "content": "Hi, how can I help you?"})
    with gr.Blocks() as app:
        with gr.Row():
            gr.Markdown("# Friday - AI")
        with gr.Row():
            with gr.Column():
                chatbot = gr.Chatbot(prompt_context.conversation)
                textfield = gr.Textbox(
                    autofocus=True,
                    placeholder="Chat with the bot!",
                    interactive=True,
                )
            with gr.Column():
                plot = gr.Plot(vector_db_tsne_figure)

        textfield.submit(
            fn=submit_message,
            inputs=[textfield],
            outputs=[textfield, chatbot, plot],
        )

    os.system("cls" if os.name == "nt" else "clear")
    app.launch()


if __name__ == "__main__":
    main()
