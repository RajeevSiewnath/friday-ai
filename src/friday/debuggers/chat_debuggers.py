import math
import os
import shutil
import sys

from colorama import Cursor, Fore, Back, Style


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
                                message["content"]
                                }"
        )
