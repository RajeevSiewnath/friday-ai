import os
from colorama import Cursor, Fore, Back, Style


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
    color_map = {"assistant": "RED", "user": "GREEN", "system": "BLUE"}
    for message in messages:
        print(
            f"{
                getattr(Back, color_map[message["role"]])
                }{
                    message["role"]
                    }{
                        Style.RESET_ALL
                        } {
                            Fore.YELLOW if message.get("status", None) == "in_progress" else getattr(Fore, color_map[message["role"]])
                            }{
                                __print_content(message["content"])
                                }"
        )
