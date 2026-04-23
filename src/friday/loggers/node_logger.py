import logging
from typing import Callable
from colorama import Fore, Back, Style
from friday.loggers.abstract_logger_config import AbstractLoggerConfig
from friday.loggers.custom_logger_adapter import CustomLoggerAdapter


class NodeLoggerFormatter(logging.Formatter):
    NODE_STR_LEN = 24
    COLOR_CYCLE = [
        "RED",
        "GREEN",
        "YELLOW",
        "BLUE",
        "MAGENTA",
        "CYAN",
    ]

    COLOR_CYCLE_INDEX = 0

    def get_next_color(self):
        color_name = NodeLoggerFormatter.COLOR_CYCLE[
            NodeLoggerFormatter.COLOR_CYCLE_INDEX
        ]
        NodeLoggerFormatter.COLOR_CYCLE_INDEX = (
            NodeLoggerFormatter.COLOR_CYCLE_INDEX + 1
        ) % len(NodeLoggerFormatter.COLOR_CYCLE)
        return getattr(Back, color_name), getattr(Fore, color_name)

    def format(self, record):
        time = f"{Style.DIM}{self.formatTime(record, '%H:%M:%S')}{Style.RESET_ALL}"

        node = getattr(record, "name", None)
        color = getattr(record, "color", None)

        if color in NodeLoggerFormatter.COLOR_CYCLE:
            color = getattr(Back, color), getattr(Fore, color)
        else:
            color = self.get_next_color()

        back, front = color
        str_len = NodeLoggerFormatter.NODE_STR_LEN
        node_part = (
            f"{back}{(node[:str_len-3]+"..."if len(node) > str_len else node):^{str_len}}{Style.RESET_ALL}"
            if node
            else ""
        )

        msg = f"{front}{record.getMessage()}{Style.RESET_ALL}"

        parts = [time, node_part, msg]
        return " | ".join(p for p in parts if p)


class NodeLoggerConfig(AbstractLoggerConfig):
    def get_formatter(self) -> logging.Formatter:
        return NodeLoggerFormatter()

    def get_adapter(
        self,
    ) -> Callable[[str], logging.LoggerAdapter]:
        def adapter(name: str) -> logging.LoggerAdapter:
            logger = logging.getLogger(name)
            return CustomLoggerAdapter(logger, "color")

        return adapter
