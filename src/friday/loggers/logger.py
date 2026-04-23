import json
import logging
import os
import sys
from typing import Any, Callable
from colorama import Fore, Style
from friday.loggers.abstract_logger_config import AbstractLoggerConfig
from friday.loggers.log_level_attributes import LogLevelAttributes
from friday.loggers.node_logger import NodeLoggerConfig
from friday.loggers.get_actual_level import get_actual_level


class BaseLogger(logging.Logger):
    def _log(self, level, msg, args, **kwargs):
        if self.isEnabledFor(level):
            args = tuple(a() if callable(a) else a for a in args)
            args = tuple(
                a if isinstance(a, str) else json.dumps(a, indent=2) for a in args
            )
        super()._log(level, msg, args, **kwargs)


logging.setLoggerClass(BaseLogger)


class DefaultFormatter(logging.Formatter):

    def format(self, record):
        level_color = LogLevelAttributes(record.levelno).color
        level = f"{level_color}{record.levelname:<8}{Style.RESET_ALL}"

        time = f"{Style.DIM}{self.formatTime(record, '%H:%M:%S')}{Style.RESET_ALL}"
        name = f"{Fore.BLUE}{record.name}{Style.RESET_ALL}"
        msg = record.getMessage()

        parts = [time, level, name, msg]
        return " | ".join(parts)


# class DefaultFilter(logging.Filter):
#     def __init__(self, name=""):
#         super().__init__(name)
#         self.filters: list[logging.Filter] = []

#     def add_filters(self, filters: list[logging.Filter]):
#         self.filters.extend(filters)

#     def filter(self, record):
#         for f in self.filters:
#             if f.filter(record):
#                 return False
#         return True


class Logger:
    __instance: "Logger" = None

    def __init__(self, loggers: dict[str, AbstractLoggerConfig], level=logging.INFO):
        if Logger.__instance is not None:
            return

        root = logging.getLogger()
        root.setLevel(get_actual_level(level))

        self.__loggers: dict[str, Callable[..., logging.LoggerAdapter]] = {}
        filters = []
        for key, logger in loggers.items():
            formatter = logger.get_formatter()
            # filter = logger.get_filter()
            adapter = logger.get_adapter()
            lvl = logger.get_level()
            self.__loggers[key] = adapter

            handler = logging.StreamHandler(sys.stdout)
            # handler.setLevel(get_actual_level(lvl))
            handler.setFormatter(formatter)
            # handler.addFilter(filter)
            filters.append(filter)
            logger_instance = logging.getLogger(key)
            logger_instance.addHandler(handler)
            logger_instance.setLevel(get_actual_level(lvl))
            logger_instance.propagate = False

        # default_filter = DefaultFilter()
        # default_filter.add_filters(filters)
        app_handler = logging.StreamHandler(sys.stdout)
        # app_handler.setLevel(level)
        app_handler.setFormatter(DefaultFormatter())
        # app_handler.addFilter(default_filter)
        root.addHandler(app_handler)
        Logger.__instance = self

    @staticmethod
    def get_logger(
        name: str,
        *args: Any,
        level: int | str = None,
    ):
        logger_key = name.split(".")[0]
        if logger_key in Logger.__instance.__loggers:
            adapter = Logger.__instance.__loggers[logger_key](name, *args)
            if level is not None:
                adapter.logger.setLevel(get_actual_level(level))
            return adapter
        else:
            logger = logging.getLogger(name)
            if level is not None:
                logger.setLevel(get_actual_level(level))
            return logger


log_level = os.getenv("LOG_LEVEL", "WARNING").upper()
node_log_level = os.getenv("NODE_LOG_LEVEL", log_level).upper()
Logger(loggers={"node": NodeLoggerConfig(level=node_log_level)}, level=log_level)

Logger.get_logger(__name__).debug("debug")
Logger.get_logger(__name__).info("info")
Logger.get_logger(__name__).warning("warning")
Logger.get_logger(__name__).error("error")
Logger.get_logger(__name__).critical("critical")
Logger.get_logger("node.debug", level="debug").debug("debug")
Logger.get_logger("node.info").info("info")
Logger.get_logger("node.warning").warning("warning")
Logger.get_logger("node.error").error("error")
Logger.get_logger("node.critical").critical("critical")
