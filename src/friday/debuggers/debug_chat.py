import os
from time import sleep
from colorama import Fore, Back, Style


def __print_content(content):
    if isinstance(content, list):
        return "\n".join([__print_content(c) for c in content])
    elif isinstance(content, dict):
        return content["text"]
    elif isinstance(content, str):
        return content
    else:
        return ""


def debug_chat(messages: list[dict]):
    os.system("cls" if os.name == "nt" else "clear")
    os.system("cls" if os.name == "nt" else "clear")
    color_map = {
        "assistant": "RED",
        "user": "GREEN",
        "system": "BLUE",
        "function_call": "MAGENTA",
        "function_call_output": "MAGENTA",
    }

    def get_agent_color(message):
        return getattr(Back, color_map[message.get("role", message.get("type"))])

    def get_agent(message):
        return message.get("role", message.get("type"))

    def get_content_color(message):
        return (
            Fore.YELLOW
            if message.get("status", None) == "in_progress"
            else getattr(Fore, color_map[message.get("role", message.get("type"))])
        )

    def get_content(message):
        return __print_content(message.get("content") or message.get("name"))

    for m in messages:
        print(
            f"{get_agent_color(m)}{get_agent(m)}{Style.RESET_ALL} {get_content_color(m)}{get_content(m)}"
        )
