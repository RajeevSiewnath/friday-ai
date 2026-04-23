import logging
from enum import Enum
from colorama import Back, Fore


class LogLevelAttributes(Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

    @property
    def level(self):
        return self.value

    @property
    def color(self):
        return {
            LogLevelAttributes.DEBUG: Fore.CYAN,
            LogLevelAttributes.INFO: Fore.GREEN,
            LogLevelAttributes.WARNING: Fore.YELLOW,
            LogLevelAttributes.ERROR: Fore.RED,
            LogLevelAttributes.CRITICAL: Back.RED,
        }[self]
