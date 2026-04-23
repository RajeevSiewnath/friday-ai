from friday.loggers.log_level_attributes import LogLevelAttributes


def get_actual_level(level):
    if isinstance(level, int):
        return level
    elif isinstance(level, str):
        try:
            return LogLevelAttributes[level.upper()].level
        except KeyError:
            raise ValueError(f"Invalid log level: {level}")
    else:
        raise ValueError(f"Invalid log level type: {type(level)}")
