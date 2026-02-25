import json
import gradio as gr
import os
from dotenv import load_dotenv
from openai import OpenAI, embeddings
from ChatLoop import ChatLoop, Role
from tools.get_sections import get_sections, get_sections_tool
from colorama import Fore, Back, Style, init
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np


init()

load_dotenv()

CONTEXT_SNIPPET = """
Context {i}:
{document}
Context {i} metadata:
{metadata}
"""

SYSTEM_PROMPT = """
You are a personal job agent for Rajeev Siewnath. 
You provide information about his curriculum vitae to the user.
If relevant, use the given context to answer any question.
{context}
"""

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
chroma = chromadb.Client(Settings(is_persistent=True))
collection: chromadb.Collection = None
chat_loop: ChatLoop = ChatLoop(
    system_prompt=SYSTEM_PROMPT.format(context=""),
    initial_history=[{"role": Role.ASSISTANT, "content": "Hi, how can I help you?"}],
    tools=[
        {
            "name": "get_sections",
            "callable": get_sections,
            "definition": get_sections_tool,
        }
    ],
)


def set_system_prompt(msg):
    context = [
        CONTEXT_SNIPPET.format(
            i=i,
            document=c["document"],
            metadata=json.dumps(c["metadata"], indent=2),
        )
        for i, c in enumerate(read_from_vector_db(msg), start=1)
    ]
    chat_loop.set_system_prompt(SYSTEM_PROMPT.format(context=context))


def submit_message(msg):
    set_system_prompt(msg)
    chat_loop.submit_message(msg)
    yield "", chat_loop.conversation, gr.Textbox(interactive=False)
    for done in chat_loop.invoke():
        yield "", chat_loop.conversation, gr.Textbox(interactive=done)


def main():
    chat_loop.reset()
    chat_loop.set_system_prompt(SYSTEM_PROMPT.format(context=""))
    chat_loop.set_initial_history(
        [{"role": Role.ASSISTANT, "content": "Hi, how can I help you?"}]
    )
    with gr.Blocks() as app:
        with gr.Row():
            with gr.Column():
                gr.Markdown("# Friday - AI")
                chatbot = gr.Chatbot(chat_loop.conversation)
                textfield = gr.Textbox(
                    autofocus=True,
                    placeholder="Chat with the bot!",
                    interactive=True,
                    elem_id="mybox",
                )
                textfield.submit(
                    fn=submit_message,
                    inputs=[textfield],
                    outputs=[textfield, chatbot, textfield],
                )
            with gr.Column():
                gr.Plot(create_scatter_plot())
                pass

    os.system("cls" if os.name == "nt" else "clear")
    app.launch()


def build_rag_vector_db():
    global collection
    folder_path = Path("data")
    files = []

    collection = chroma.get_collection(name="cv-rajeev-siewnath")

    if not collection:
        for file_path in folder_path.rglob("*"):
            if file_path.is_file():
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                files.append(
                    {
                        "file_source": str(file_path.relative_to(folder_path)),
                        "doc_type": str(file_path.parent.relative_to(folder_path)),
                        "content": json.loads(content),
                    }
                )

        embeddings_result = embeddings.create(
            model="text-embedding-3-small",
            input=[file["content"]["document"] for file in files],
        ).data
        vectors = [e.embedding for e in embeddings_result]

        collection = chroma.create_collection(name="cv-rajeev-siewnath")
        collection.add(
            ids=[file["content"]["id"] for file in files],
            embeddings=vectors,
            metadatas=[
                {
                    **file["content"]["metadata"],
                    "doc_type": file["doc_type"],
                    "file_source": file["file_source"],
                }
                for file in files
            ],
            documents=[file["content"]["document"] for file in files],
        )

        print("Skills entry ingested into Chroma successfully!")
    else:
        print("Chroma db is probably existing already")


def read_from_vector_db(query):
    global collection
    results = collection.query(query_texts=query, n_results=3)
    return [
        {"document": result[0], "id": result[1], "metadata": result[2]}
        for result in zip(
            results["documents"][0], results["ids"][0], results["metadatas"][0]
        )
    ]


def create_scatter_plot():
    global collection
    types = [metadata["type"] for metadata in collection.get()["metadatas"]]
    colors = [
        ["blue", "green", "red", "orange", "purple"][
            ["education", "project", "profile", "skills", "employment"].index(t)
        ]
        for t in types
    ]
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    reduced_vectors = tsne.fit_transform(
        np.array(collection.get(include=["embeddings"])["embeddings"])
    )

    # Create the 2D scatter plot
    fig = go.Figure(
        data=[
            go.Scatter(
                x=reduced_vectors[:, 0],
                y=reduced_vectors[:, 1],
                mode="markers",
                marker=dict(size=5, color=colors, opacity=0.8),
                text=[
                    f"Type: {t}<br>Text: {d[:100]}..."
                    for t, d in zip(
                        types, collection.get(include=["documents"])["documents"]
                    )
                ],
                hoverinfo="text",
            )
        ]
    )

    fig.update_layout(
        title="2D Chroma Vector Store Visualization",
        scene=dict(xaxis_title="x", yaxis_title="y"),
        margin=dict(r=20, b=10, l=10, t=40),
    )

    return fig


if __name__ == "__main__":
    print("Running!")
    build_rag_vector_db()
    main()
