import logging
from enum import Enum
from colorama import Back, Fore


class LogLevelAttributes(Enum):
    TRACE = 5
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    NOTICE = 25
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

    @property
    def level(self):
        return self.value

    @property
    def color(self):
        return {
            LogLevelAttributes.TRACE: Fore.MAGENTA,
            LogLevelAttributes.DEBUG: Fore.CYAN,
            LogLevelAttributes.INFO: Fore.GREEN,
            LogLevelAttributes.NOTICE: Fore.YELLOW,
            LogLevelAttributes.WARNING: Back.YELLOW,
            LogLevelAttributes.ERROR: Fore.RED,
            LogLevelAttributes.CRITICAL: Back.RED,
        }[self]
