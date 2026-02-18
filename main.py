import json
from pprint import pprint
import gradio as gr
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

history = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "assistant", "content": "Hi, how can I help you?"},
]


def submit_message(msg):
    history.append({"role": "user", "content": msg})
    yield "", history, gr.Textbox(interactive=False)

    history.append({"role": "assistant", "content": ""})

    with client.responses.stream(model="gpt-4.1-nano", input=history) as stream:
        for event in stream:
            if event.type == "response.output_text.delta":
                history[-1]["content"] += event.delta
                yield "", history, gr.Textbox(interactive=False)

            elif event.type == "response.completed":
                print(event.model_dump_json(indent=2))
                yield "", history, gr.Textbox(interactive=True)


def main():
    with gr.Blocks() as app:
        with gr.Row():
            with gr.Column():
                gr.Markdown("# Friday - AI")
                chatbot = gr.Chatbot(history)
                textfield = gr.Textbox(
                    autofocus=True, placeholder="Chat with the bot!", interactive=True
                )
                textfield.submit(
                    fn=submit_message,
                    inputs=[textfield],
                    outputs=[textfield, chatbot, textfield],
                )
            with gr.Column():
                pass

    app.launch()


if __name__ == "__main__":
    main()
