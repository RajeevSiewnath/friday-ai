from typing import Annotated, Any, TypedDict
from friday.graph.evaluation.reducers.scores_reducer import scores_reducer


class ScoresState(TypedDict):
    evaluation_scores: Annotated[dict[str, list[Any]], scores_reducer]
