import logging
from friday.loggers.log_level_attributes import LogLevelAttributes


class CustomLoggerAdapter(logging.LoggerAdapter):
    def __init__(self, logger, *keys: str, extra=None, merge_extra=False):
        super().__init__(logger, extra=extra, merge_extra=merge_extra)
        self.keys = keys

    def trace(self, msg, *args, **kwargs):
        self.log(LogLevelAttributes.TRACE.level, msg, *args, **kwargs)

    def notice(self, msg, *args, **kwargs):
        self.log(LogLevelAttributes.NOTICE.level, msg, *args, **kwargs)

    def process(self, msg, kwargs):
        extra = kwargs.setdefault("extra", {})
        if self.extra:
            for key in self.keys:
                if key in self.extra:
                    extra[key] = self.extra[key]
        return msg, kwargs
