from typing import TypedDict


class StatisticsState(TypedDict):
    data_set_statistics: dict[str, float]
