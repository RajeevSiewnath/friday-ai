import pytest


@pytest.fixture
def scores_dict_1():
    return {
        "accuracy": [0.85, 0.90],
        "completeness": [0.80, 0.85],
    }


@pytest.fixture
def scores_dict_2():
    return {
        "accuracy": [0.88],
        "relevance": [0.92],
    }


@pytest.fixture
def empty_scores_dict():
    return {}


@pytest.fixture
def single_metric_dict():
    return {"metric": [0.75]}
