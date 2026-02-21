import json
from pprint import pprint
import gradio as gr
import os
from dotenv import load_dotenv
from openai import OpenAI
from tools.get_sections import get_sections, get_sections_tool
from tools.get_section import get_section, get_section_tool
from colorama import Fore, Back, Style, init

init()

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

history = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "assistant", "content": "Hi, how can I help you?"},
]


def handle_function_call(stream_id, call_id, name, args):
    tools_map = {
        "get_sections": get_sections,
        "get_section": get_section,
    }

    func = tools_map[name]
    result = func(**args)
    history.append({"role": "assistant", "content": ""})

    with client.responses.stream(
        model="gpt-4.1-nano",
        tools=[get_section_tool, get_sections_tool],
        input=history
        + [{"type": "function_call_output", "call_id": call_id, "output": str(result)}],
        previous_response_id=stream_id,
    ) as stream:
        for event in stream:
            print(Fore.GREEN + event.model_dump_json(indent=2) + Style.RESET_ALL)
            if event.type == "response.output_text.delta":
                history[-1]["content"] += event.delta
                print(Back.RED + json.dumps(history, indent=2) + Style.RESET_ALL)
                yield
            else:
                pass


def submit_message(msg):
    history.append({"role": "user", "content": msg})
    yield "", history, gr.Textbox(interactive=False)

    history.append({"role": "assistant", "content": ""})
    callables = {}

    with client.responses.stream(
        model="gpt-4.1-nano", tools=[get_section_tool, get_sections_tool], input=history
    ) as stream:
        print("STREAM")
        for event in stream:
            print("EVENT")
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
                print(Back.RED + json.dumps(history, indent=2) + Style.RESET_ALL)
                for _ in handle_function_call(stream_id, call_id, name, args):
                    pass
                yield "", history, gr.Textbox(interactive=True, autofocus=True)
            elif event.type == "response.output_text.delta":
                history[-1]["content"] += event.delta
                print(Back.RED + json.dumps(history, indent=2) + Style.RESET_ALL)
                yield "", history, gr.Textbox(interactive=True, autofocus=True)
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


if __name__ == "__main__":
    main()
