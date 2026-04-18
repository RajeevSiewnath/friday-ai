import pytest
from typing import get_type_hints, get_origin, get_args
from friday.graph.query.states.system_prompt_state import SystemPromptState
from friday.graph.query.reducers.system_prompt_reducer import (
    system_prompt_reducer,
    SystemPromptReducerResetAction,
)


class TestSystemPromptState:
    def test_state_has_system_prompt_field(self):
        """Test that SystemPromptState has the system_prompt field."""
        assert "system_prompt" in SystemPromptState.__annotations__

    def test_system_prompt_field_is_string(self):
        """Test that system_prompt field is annotated as string."""
        hints = get_type_hints(SystemPromptState, include_extras=True)
        annotation = SystemPromptState.__annotations__["system_prompt"]

        # Should be Annotated type
        origin = get_origin(annotation)
        assert origin is not None

    def test_state_is_typed_dict(self):
        """Test that SystemPromptState is a TypedDict."""
        assert hasattr(SystemPromptState, "__annotations__")
        assert isinstance(SystemPromptState.__annotations__, dict)

    def test_reducer_appends_to_string(self):
        """Test that system_prompt_reducer appends strings with separator."""
        result = system_prompt_reducer("prompt1", "prompt2")
        assert result == "prompt1\n\nprompt2"

    def test_reducer_with_reset_action(self):
        """Test that system_prompt_reducer replaces on reset action."""
        action = SystemPromptReducerResetAction("new_prompt")
        result = system_prompt_reducer("old_prompt", action)
        assert result == "new_prompt"

    def test_reducer_reset_overwrites_previous(self):
        """Test that reset action completely overwrites previous content."""
        old_content = "line1\nline2\nline3"
        action = SystemPromptReducerResetAction("completely_new")
        result = system_prompt_reducer(old_content, action)
        assert result == "completely_new"

    def test_reducer_multiple_appends(self):
        """Test multiple sequential appends."""
        result = system_prompt_reducer("p1", "p2")
        result = system_prompt_reducer(result, "p3")
        result = system_prompt_reducer(result, "p4")

        assert "p1" in result
        assert "p2" in result
        assert "p3" in result
        assert "p4" in result
        assert result.count("\n\n") == 3  # 3 separators for 4 items

    def test_reset_action_initialization(self):
        """Test SystemPromptReducerResetAction initialization."""
        content = "test content"
        action = SystemPromptReducerResetAction(content)
        assert action.content == content

    def test_reset_action_with_empty_string(self):
        """Test reset action with empty string."""
        action = SystemPromptReducerResetAction("")
        result = system_prompt_reducer("old", action)
        assert result == ""

    def test_reset_action_with_multiline_content(self):
        """Test reset action with multiline content."""
        multiline = "line1\nline2\nline3"
        action = SystemPromptReducerResetAction(multiline)
        result = system_prompt_reducer("old", action)
        assert result == multiline

    def test_reducer_with_special_characters(self):
        """Test reducer with special characters."""
        result = system_prompt_reducer("prompt@#$", "prompt%^&")
        assert "@#$" in result
        assert "%^&" in result

    def test_reducer_preserves_whitespace(self):
        """Test that reducer preserves whitespace in prompts."""
        result = system_prompt_reducer("  spaced  ", "  also spaced  ")
        assert "  spaced  " in result
        assert "  also spaced  " in result

    def test_state_instance_creation(self):
        """Test that we can create instances of SystemPromptState."""
        state = SystemPromptState(system_prompt="test prompt")
        assert state["system_prompt"] == "test prompt"

    def test_state_with_empty_prompt(self):
        """Test state with empty prompt."""
        state = SystemPromptState(system_prompt="")
        assert state["system_prompt"] == ""

    def test_state_with_multiline_prompt(self):
        """Test state with multiline prompt."""
        multiline = "line1\nline2\nline3"
        state = SystemPromptState(system_prompt=multiline)
        assert state["system_prompt"] == multiline

    def test_reducer_separator_format(self):
        """Test that separator is specifically double newline."""
        result = system_prompt_reducer("a", "b")
        # Should have exactly two newlines between prompts
        assert "\n\n" in result
        assert result == "a\n\nb"
