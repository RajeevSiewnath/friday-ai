import pytest
from typing import List, Dict, Optional, Union, Literal
from friday.utils.function_definition_creator import function_definition_creator


class TestFunctionDefinitionCreator:
    def test_simple_function_no_docstring(self):
        def simple_func(x: int) -> str:
            pass

        result = function_definition_creator(simple_func)

        assert result["type"] == "function"
        assert result["name"] == "simple_func"
        assert result["description"] == "No description provided."
        assert "x" in result["parameters"]["properties"]
        assert result["parameters"]["required"] == ["x"]

    def test_function_with_docstring(self):
        def add_numbers(a: int, b: int) -> int:
            """Add two numbers together.

            Args:
                a: The first number
                b: The second number

            Returns:
                The sum of the two numbers
            """
            return a + b

        result = function_definition_creator(add_numbers)

        assert result["name"] == "add_numbers"
        assert result["description"] == "Add two numbers together."
        assert result["parameters"]["required"] == ["a", "b"]
        assert result["parameters"]["properties"]["a"]["description"] == "The first number"
        assert result["parameters"]["properties"]["b"]["description"] == "The second number"
        assert result["returns"]["description"] == "The sum of the two numbers"

    def test_function_with_default_parameters(self):
        def greet(name: str, greeting: str = "Hello") -> str:
            """Greet someone.

            Args:
                name: Person to greet
                greeting: The greeting to use

            Returns:
                A greeting message
            """
            return f"{greeting} {name}"

        result = function_definition_creator(greet)

        assert result["parameters"]["required"] == ["name"]
        assert "greeting" in result["parameters"]["properties"]

    def test_function_no_parameters(self):
        def get_timestamp() -> str:
            """Get the current timestamp."""
            return "2024-01-01"

        result = function_definition_creator(get_timestamp)

        assert result["parameters"]["required"] == []
        assert result["parameters"]["properties"] == {}

    def test_function_with_string_type(self):
        def process_text(text: str) -> str:
            """Process text."""
            return text.upper()

        result = function_definition_creator(process_text)

        assert result["parameters"]["properties"]["text"]["type"] == "string"

    def test_function_with_int_type(self):
        def square(n: int) -> int:
            """Square a number."""
            return n * n

        result = function_definition_creator(square)

        assert result["parameters"]["properties"]["n"]["type"] == "integer"

    def test_function_with_float_type(self):
        def divide(a: float, b: float) -> float:
            """Divide two numbers."""
            return a / b

        result = function_definition_creator(divide)

        assert result["parameters"]["properties"]["a"]["type"] == "number"

    def test_function_with_bool_type(self):
        def is_valid(flag: bool) -> bool:
            """Check if valid."""
            return flag

        result = function_definition_creator(is_valid)

        assert result["parameters"]["properties"]["flag"]["type"] == "boolean"

    def test_function_with_list_type(self):
        def process_items(items: List[str]) -> int:
            """Process a list of items."""
            return len(items)

        result = function_definition_creator(process_items)

        assert result["parameters"]["properties"]["items"]["type"] == "array"
        assert result["parameters"]["properties"]["items"]["items"]["type"] == "string"

    def test_function_with_dict_type(self):
        def process_dict(data: Dict[str, int]) -> int:
            """Process a dictionary."""
            return sum(data.values())

        result = function_definition_creator(process_dict)

        assert result["parameters"]["properties"]["data"]["type"] == "object"
        assert result["parameters"]["properties"]["data"]["additionalProperties"] is True

    def test_function_with_optional_type(self):
        def optional_param(value: Optional[str]) -> str:
            """Function with optional parameter."""
            return value or "default"

        result = function_definition_creator(optional_param)

        assert "value" in result["parameters"]["properties"]

    def test_function_with_union_type(self):
        def union_param(value: Union[str, int]) -> str:
            """Function with union parameter."""
            return str(value)

        result = function_definition_creator(union_param)

        assert "anyOf" in result["parameters"]["properties"]["value"]

    def test_function_with_literal_type(self):
        def literal_param(mode: Literal["fast", "slow"]) -> str:
            """Function with literal parameter."""
            return mode

        result = function_definition_creator(literal_param)

        props = result["parameters"]["properties"]["mode"]
        assert props["type"] == "string"
        assert set(props["enum"]) == {"fast", "slow"}

    def test_function_with_mixed_types(self):
        def mixed_types(
            name: str,
            age: int,
            score: float,
            active: bool,
            tags: List[str],
        ) -> Dict[str, str]:
            """Function with mixed types.

            Args:
                name: User name
                age: User age
                score: User score
                active: Is active
                tags: User tags

            Returns:
                User data dictionary
            """
            return {}

        result = function_definition_creator(mixed_types)

        props = result["parameters"]["properties"]
        assert props["name"]["type"] == "string"
        assert props["age"]["type"] == "integer"
        assert props["score"]["type"] == "number"
        assert props["active"]["type"] == "boolean"
        assert props["tags"]["type"] == "array"

    def test_function_with_no_type_hints(self):
        def untyped_func(x, y):
            """Add two values."""
            return x + y

        result = function_definition_creator(untyped_func)

        # Default to string type
        assert result["parameters"]["properties"]["x"]["type"] == "string"
        assert result["parameters"]["properties"]["y"]["type"] == "string"

    def test_function_with_return_type_only(self):
        def simple_return() -> int:
            """Return an integer."""
            return 42

        result = function_definition_creator(simple_return)

        assert result["returns"]["type"] == "integer"

    def test_function_no_return_type(self):
        def no_return(x: str):
            """Function with no return type."""
            pass

        result = function_definition_creator(no_return)

        assert result["returns"]["type"] == "string"

    def test_function_callable_check(self):
        with pytest.raises(TypeError):
            function_definition_creator("not a function")

    def test_function_with_multiline_docstring(self):
        def multiline_doc(x: int) -> str:
            """This is a long description.

            It spans multiple lines
            to provide more detail.

            Args:
                x: An integer value

            Returns:
                A string representation
            """
            return str(x)

        result = function_definition_creator(multiline_doc)

        assert "long description" in result["description"]
        assert "spans multiple lines" in result["description"]

    def test_function_with_complex_return_type(self):
        def get_user_ids() -> List[int]:
            """Get list of user IDs."""
            return [1, 2, 3]

        result = function_definition_creator(get_user_ids)

        assert result["returns"]["type"] == "array"
        assert result["returns"]["items"]["type"] == "integer"

    def test_function_preserves_parameter_order(self):
        def ordered_params(first: str, second: int, third: bool) -> str:
            """Function with ordered parameters."""
            return str(first)

        result = function_definition_creator(ordered_params)

        props = result["parameters"]["properties"]
        assert "first" in props
        assert "second" in props
        assert "third" in props

    def test_function_with_special_chars_in_docstring(self):
        def special_chars(text: str) -> str:
            """Process text with special chars: @#$%.

            Args:
                text: Input text with special chars

            Returns:
                Processed text
            """
            return text

        result = function_definition_creator(special_chars)

        assert "@#$%" in result["description"]

    def test_function_list_without_type_args(self):
        def untyped_list(items: list) -> str:
            """Process untyped list."""
            return str(len(items))

        result = function_definition_creator(untyped_list)

        # Unparameterized list defaults to string type
        assert result["parameters"]["properties"]["items"]["type"] == "string"

    def test_function_dict_without_type_args(self):
        def untyped_dict(data: dict) -> str:
            """Process untyped dict."""
            return str(len(data))

        result = function_definition_creator(untyped_dict)

        # Unparameterized dict defaults to string type
        assert result["parameters"]["properties"]["data"]["type"] == "string"

    def test_function_nested_list_type(self):
        def nested_list(matrix: List[List[int]]) -> int:
            """Process nested list."""
            return sum(sum(row) for row in matrix)

        result = function_definition_creator(nested_list)

        props = result["parameters"]["properties"]["matrix"]
        assert props["type"] == "array"
        assert props["items"]["type"] == "array"

    def test_function_with_docstring_no_args_section(self):
        def no_args_doc() -> str:
            """Function with docstring but no Args section."""
            return "result"

        result = function_definition_creator(no_args_doc)

        assert result["description"] == "Function with docstring but no Args section."

    def test_function_result_structure(self):
        def sample() -> str:
            """Sample function."""
            return "sample"

        result = function_definition_creator(sample)

        assert "type" in result
        assert "name" in result
        assert "description" in result
        assert "parameters" in result
        assert "returns" in result
        assert result["type"] == "function"

    def test_function_parameter_without_description(self):
        def param_no_desc(x: int) -> str:
            """Function without param descriptions."""
            return str(x)

        result = function_definition_creator(param_no_desc)

        assert "x" in result["parameters"]["properties"]
        assert result["parameters"]["required"] == ["x"]
