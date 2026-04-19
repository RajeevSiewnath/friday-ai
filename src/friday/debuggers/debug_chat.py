import os
from colorama import Fore, Back, Style
from friday.utils.print_content import print_content


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
        return print_content(message.get("content") or message.get("name"))

    for m in messages:
        print(
            f"{get_agent_color(m)}{get_agent(m)}{Style.RESET_ALL} {get_content_color(m)}{get_content(m)}"
        )
