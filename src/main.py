import os
from core.chat_loop import ChatLoop
from core.llm import LLM
from core.prompt_context import PromptContext
import gradio as gr
from core.vector_db import VectorDB
from models.query_context import QueryContext
from pipelines.pipeline_factory import PipelineFactory
from query_pipes.rag_context_injector import RagContextInjector
from query_pipes.rag_context_retriever import RagContextRetriever
from query_pipes.rag_tsne_vis_updater import RagTSNEVisUpdater
from visualizations.vector_db_tsne_visualization import VectorDBTSNEVisualization

llm = LLM()
prompt_context = PromptContext(
    user_context="""
You are a personal job agent for Rajeev Siewnath. 
You provide information about his curriculum vitae to the user, who is a person interested in Rajeev Siewnath's career.
If relevant, use the given context to answer any question.
"""
)
vector_db = VectorDB(llm=llm, name="cv-rajeev-siewnath")
chat_loop = ChatLoop(llm=llm, prompt_context=prompt_context)

vector_db_tsne_visualization = VectorDBTSNEVisualization(
    title="tSNE", collection=vector_db.collection
)

pipeline_factory = PipelineFactory(
    llm=llm,
    prompt_context=prompt_context,
    vector_db=vector_db,
)

query_pipeline = pipeline_factory.make(
    RagContextRetriever(n_results=5),
    RagContextInjector(),
    RagTSNEVisUpdater(vector_db_tsne_visualization),
)

vector_db_tsne_figure = vector_db_tsne_visualization.get()


def submit_message(question):
    if not chat_loop.is_looping and question:
        query_context = query_pipeline.run(QueryContext(question=question))
        prompt_context.push({"role": "user", "content": query_context.question})
        vector_db_tsne_figure = vector_db_tsne_visualization.get()
        yield gr.Textbox(
            value="", placeholder="Working..."
        ), prompt_context.conversation, vector_db_tsne_figure
        for done in chat_loop.invoke():
            yield gr.Textbox(
                value="", placeholder="Working..." if not done else "Chat with the bot!"
            ), prompt_context.conversation, vector_db_tsne_figure
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
