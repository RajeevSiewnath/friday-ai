import json
from pprint import pprint
import gradio as gr
import os
from dotenv import load_dotenv
from openai import OpenAI
from tools.get_sections import get_sections, get_sections_tool
from tools.get_section import get_section, get_section_tool

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

history = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "assistant", "content": "Hi, how can I help you?"},
]


def handle_function_call(name, args):
    tools_map = {
        "get_sections": get_sections,
        "get_section": get_section,
    }

    func = tools_map[name]
    result = func(**args)
    # client.responses.


def submit_message(msg):
    history.append({"role": "user", "content": msg})
    yield "", history, gr.Textbox(interactive=False)

    history.append({"role": "assistant", "content": ""})

    with client.responses.stream(
        model="gpt-4.1-nano", tools=[get_section_tool, get_sections_tool], input=history
    ) as stream:
        for event in stream:
            print(event.model_dump_json(indent=2))
            print("#####")
            if event.type == "response.output_text.delta":
                history[-1]["content"] += event.delta
            elif event.type == "response.completed":
                for output in event.response.output:
                    if output.type == "function_call":
                        handle_function_call(output.name, output.arguments)
                else:
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
