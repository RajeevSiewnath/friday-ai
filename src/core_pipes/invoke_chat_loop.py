from typing import Any, Generator
from core.chat_loop import ChatLoop
from pipelines.pipeline import Pipe


class InvokeChatLoop(Pipe[Any, Generator[bool, None, None]]):

    def __init__(self, chat_loop: ChatLoop):
        super().__init__()
        self.chat_loop = chat_loop

    def run(self, input):
        return self.chat_loop.invoke(prompt_context=self.prompt_context)
