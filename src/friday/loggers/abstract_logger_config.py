from abc import ABC, abstractmethod
from typing import Callable
import logging


class AbstractLoggerConfig(ABC):

    def __init__(self, level=logging.INFO):
        self.level = level

    def get_level(self):
        return self.level

    @abstractmethod
    def get_formatter(self) -> logging.Formatter:
        raise NotImplementedError

    # @abstractmethod
    # def get_filter(self) -> logging.Filter:
    #     raise NotImplementedError

    @abstractmethod
    def get_adapter(
        self,
    ) -> Callable[[str], logging.LoggerAdapter]:
        raise NotImplementedError
