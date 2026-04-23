import logging


class CustomLoggerAdapter(logging.LoggerAdapter):
    def __init__(self, logger, *keys: str, extra=None, merge_extra=False):
        super().__init__(logger, extra=extra, merge_extra=merge_extra)
        self.keys = keys

    def process(self, msg, kwargs):
        extra = kwargs.setdefault("extra", {})
        if self.extra:
            for key in self.keys:
                if key in self.extra:
                    extra[key] = self.extra[key]
        return msg, kwargs
