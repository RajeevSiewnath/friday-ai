import pytest
import tempfile
import json
import os
from pathlib import Path
from friday.core.evaluation import Evaluation
from friday.graph.training.data_sets_loader_factory import data_sets_loader_factory


class TestDataSetsLoaderFactory:
    def test_loader_factory_returns_callable(self, evaluation_json_file):
        loader = data_sets_loader_factory(evaluation_json_file)
        assert callable(loader)

    def test_loader_returns_all_required_keys(self, evaluation_json_file):
        loader = data_sets_loader_factory(evaluation_json_file)
        result = loader()
        assert "full_data_set" in result
        assert "training_data_set" in result
        assert "validating_data_set" in result
        assert "testing_data_set" in result

    def test_loader_returns_lists(self, evaluation_json_file):
        loader = data_sets_loader_factory(evaluation_json_file)
        result = loader()
        assert isinstance(result["full_data_set"], list)
        assert isinstance(result["training_data_set"], list)
        assert isinstance(result["validating_data_set"], list)
        assert isinstance(result["testing_data_set"], list)

    def test_loader_returns_evaluation_objects(self, evaluation_json_file):
        loader = data_sets_loader_factory(evaluation_json_file)
        result = loader()
        for key in ["full_data_set", "training_data_set", "validating_data_set", "testing_data_set"]:
            for item in result[key]:
                assert isinstance(item, Evaluation)

    def test_loader_respects_max_parameter(self, evaluation_json_file):
        max_items = 2
        loader = data_sets_loader_factory(evaluation_json_file, max=max_items)
        result = loader()
        assert len(result["full_data_set"]) == max_items

    def test_loader_without_max_loads_all(self, evaluation_json_file):
        loader = data_sets_loader_factory(evaluation_json_file)
        result = loader()
        assert len(result["full_data_set"]) == 5  # sample_evaluations has 5 items

    def test_loader_splits_data_with_default_ratios(self, evaluation_json_file):
        loader = data_sets_loader_factory(evaluation_json_file)
        result = loader()
        total = (
            len(result["training_data_set"])
            + len(result["validating_data_set"])
            + len(result["testing_data_set"])
        )
        assert total == len(result["full_data_set"])

    def test_loader_respects_custom_ratios(self, evaluation_json_file):
        training_ratio = 5
        validating_ratio = 3
        testing_ratio = 2
        loader = data_sets_loader_factory(
            evaluation_json_file,
            training_ratio=training_ratio,
            validating_ratio=validating_ratio,
            testing_ratio=testing_ratio,
        )
        result = loader()
        total = (
            len(result["training_data_set"])
            + len(result["validating_data_set"])
            + len(result["testing_data_set"])
        )
        assert total == len(result["full_data_set"])

    def test_loader_training_set_largest_with_default_ratios(self, evaluation_json_file):
        loader = data_sets_loader_factory(evaluation_json_file)
        result = loader()
        assert len(result["training_data_set"]) >= len(result["validating_data_set"])
        assert len(result["training_data_set"]) >= len(result["testing_data_set"])

    def test_loader_with_custom_path(self, sample_evaluations):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test_data.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump([e.model_dump() for e in sample_evaluations], f)

            loader = data_sets_loader_factory(file_path)
            result = loader()
            assert len(result["full_data_set"]) == len(sample_evaluations)

    def test_loader_creates_evaluation_with_all_fields(self, evaluation_json_file):
        loader = data_sets_loader_factory(evaluation_json_file)
        result = loader()
        evaluation = result["full_data_set"][0]
        assert hasattr(evaluation, "question")
        assert hasattr(evaluation, "answer")
        assert hasattr(evaluation, "category")
        assert hasattr(evaluation, "keywords")

    def test_loader_data_not_duplicated_across_sets(self, evaluation_json_file):
        loader = data_sets_loader_factory(evaluation_json_file)
        result = loader()

        training = set(e.question for e in result["training_data_set"])
        validating = set(e.question for e in result["validating_data_set"])
        testing = set(e.question for e in result["testing_data_set"])

        assert len(training & validating) == 0
        assert len(training & testing) == 0
        assert len(validating & testing) == 0

    def test_loader_max_zero(self, evaluation_json_file):
        loader = data_sets_loader_factory(evaluation_json_file, max=0)
        result = loader()
        assert len(result["full_data_set"]) == 0
        assert len(result["training_data_set"]) == 0

    def test_loader_max_exceeds_file_size(self, evaluation_json_file):
        loader = data_sets_loader_factory(evaluation_json_file, max=1000)
        result = loader()
        assert len(result["full_data_set"]) == 5

    def test_loader_callable_multiple_times(self, evaluation_json_file):
        loader = data_sets_loader_factory(evaluation_json_file)
        result1 = loader()
        result2 = loader()

        assert len(result1["full_data_set"]) == len(result2["full_data_set"])
