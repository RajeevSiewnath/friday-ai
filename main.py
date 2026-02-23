import json
import gradio as gr
import os
from dotenv import load_dotenv
from openai import OpenAI
from tools.get_sections import get_sections, get_sections_tool
from colorama import Fore, Back, Style, init
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings


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

history = [
    {"role": "system", "content": SYSTEM_PROMPT.format(context="")},
    {"role": "assistant", "content": "Hi, how can I help you?"},
]


def set_system_prompt(msg):
    context = [
        CONTEXT_SNIPPET.format(
            i=i,
            document=c["document"],
            metadata=json.dumps(c["metadata"], indent=2),
        )
        for i, c in enumerate(read_from_vector_db(msg), start=1)
    ]
    history[0]["content"] = SYSTEM_PROMPT.format(context=context)


def handle_function_call(stream_id, call_id, name, args):
    tools_map = {"get_sections": get_sections}

    func = tools_map[name]
    result = func(**args)

    with client.responses.stream(
        model="gpt-4.1-nano",
        tools=[get_sections_tool],
        input=history
        + [{"type": "function_call_output", "call_id": call_id, "output": str(result)}],
        previous_response_id=stream_id,
    ) as stream:
        for event in stream:
            print(Fore.GREEN + event.model_dump_json(indent=2) + Style.RESET_ALL)
            if event.type == "response.output_text.delta":
                history[-1]["content"] += event.delta
                # print(Back.YELLOW + json.dumps(history, indent=2) + Style.RESET_ALL)
                yield
            elif event.type == "response.completed":
                print(Back.YELLOW + json.dumps(history, indent=2) + Style.RESET_ALL)
            else:
                pass


def submit_message(msg):
    set_system_prompt(msg)
    history.append({"role": "user", "content": msg})
    yield "", history, gr.Textbox(interactive=False)

    history.append({"role": "assistant", "content": ""})
    callables = {}

    with client.responses.stream(
        model="gpt-4.1-nano",
        tools=[get_sections_tool],
        input=history,
    ) as stream:
        for event in stream:
            if event.type == "response.output_item.added":
                color = Fore.CYAN
            elif event.type == "response.function_call_arguments.delta":
                color = Fore.LIGHTCYAN_EX
            elif event.type == "response.function_call_arguments.done":
                color = Back.CYAN
            elif event.type == "response.output_text.delta":
                color = Fore.BLUE
            else:
                color = Back.RED

            print(color + event.model_dump_json(indent=2) + Style.RESET_ALL)
            print(Fore.RED + "#############################" + Style.RESET_ALL)

            if event.type == "response.output_item.added":
                if event.item.type == "function_call":
                    callables[event.item.id] = {
                        "call_id": event.item.call_id,
                        "name": event.item.name,
                        "args": "",
                    }
            elif event.type == "response.function_call_arguments.delta":
                callables[event.item_id]["args"] += event.delta
            elif event.type == "response.function_call_arguments.done":
                stream_id = stream.get_final_response().id
                call_id = callables[event.item_id]["call_id"]
                name = callables[event.item_id]["name"]
                args = json.loads(callables[event.item_id]["args"])
                for _ in handle_function_call(stream_id, call_id, name, args):
                    pass
                yield "", history, gr.Textbox(interactive=True, autofocus=True)
            elif event.type == "response.output_text.delta":
                history[-1]["content"] += event.delta
                yield "", history, gr.Textbox(interactive=True, autofocus=True)
            elif event.type == "response.completed":
                print(Back.YELLOW + json.dumps(history, indent=2) + Style.RESET_ALL)
            else:
                pass


def main():
    with gr.Blocks() as app:
        with gr.Row():
            with gr.Column():
                gr.Markdown("# Friday - AI")
                chatbot = gr.Chatbot(history)
                textfield = gr.Textbox(
                    value="which sections?",
                    autofocus=True,
                    placeholder="Chat with the bot!",
                    interactive=True,
                )
                textfield.submit(
                    fn=submit_message,
                    inputs=[textfield],
                    outputs=[textfield, chatbot, textfield],
                )
            with gr.Column():
                pass

    os.system("cls" if os.name == "nt" else "clear")
    app.launch()


def build_rag_vector_db():
    global collection
    folder_path = Path("data")
    files = []

    embedding_function = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ.get("OPENAI_API_KEY"), model_name="text-embedding-3-small"
    )

    try:
        collection = chroma.create_collection(
            name="cv-rajeev-siewnath", embedding_function=embedding_function
        )

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

        collection.add(
            ids=[file["content"]["id"] for file in files],
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
    except:
        collection = chroma.get_collection(name="cv-rajeev-siewnath")
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


if __name__ == "__main__":
    print("Running!")
    build_rag_vector_db()
    print(collection)
    main()
