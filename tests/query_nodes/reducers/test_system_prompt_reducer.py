import pytest
from friday.query_nodes.reducers.system_prompt_reducer import (
    system_prompt_reducer,
    SystemPromptReducerResetAction,
)


class TestSystemPromptReducer:
    def test_reducer_appends_strings(self):
        """Test that reducer appends strings with newline separator."""
        result = system_prompt_reducer("prompt1", "prompt2")
        assert result == "prompt1\n\nprompt2"

    def test_reducer_with_empty_left(self):
        """Test reducer with empty left string."""
        result = system_prompt_reducer("", "prompt")
        assert result == "\n\nprompt"

    def test_reducer_with_empty_right(self):
        """Test reducer with empty right string."""
        result = system_prompt_reducer("prompt", "")
        assert result == "prompt\n\n"

    def test_reducer_with_both_empty(self):
        """Test reducer with both strings empty."""
        result = system_prompt_reducer("", "")
        assert result == "\n\n"

    def test_reducer_with_multiline_left(self):
        """Test reducer with multiline left string."""
        left = "line1\nline2\nline3"
        right = "additional"
        result = system_prompt_reducer(left, right)
        assert result == "line1\nline2\nline3\n\nadditional"

    def test_reducer_with_multiline_right(self):
        """Test reducer with multiline right string."""
        left = "initial"
        right = "line1\nline2\nline3"
        result = system_prompt_reducer(left, right)
        assert result == "initial\n\nline1\nline2\nline3"

    def test_reducer_with_both_multiline(self):
        """Test reducer with both multiline strings."""
        left = "start\nline2"
        right = "continue\nline4"
        result = system_prompt_reducer(left, right)
        assert result == "start\nline2\n\ncontinue\nline4"

    def test_reducer_with_special_characters(self):
        """Test reducer with special characters."""
        result = system_prompt_reducer("prompt@#$", "prompt%^&")
        assert "@#$" in result
        assert "%^&" in result
        assert "\n\n" in result

    def test_reducer_with_reset_action(self):
        """Test reducer with reset action replaces content."""
        old = "old_prompt"
        action = SystemPromptReducerResetAction("new_prompt")
        result = system_prompt_reducer(old, action)
        assert result == "new_prompt"

    def test_reducer_reset_ignores_left(self):
        """Test that reset action completely ignores left value."""
        old = "very long old prompt\nwith multiple lines\nand content"
        action = SystemPromptReducerResetAction("x")
        result = system_prompt_reducer(old, action)
        assert result == "x"

    def test_reducer_reset_with_empty_string(self):
        """Test reset action with empty string."""
        action = SystemPromptReducerResetAction("")
        result = system_prompt_reducer("old", action)
        assert result == ""

    def test_reducer_reset_with_multiline_content(self):
        """Test reset action with multiline content."""
        multiline = "new line1\nnew line2\nnew line3"
        action = SystemPromptReducerResetAction(multiline)
        result = system_prompt_reducer("old", action)
        assert result == multiline

    def test_reset_action_initialization(self):
        """Test SystemPromptReducerResetAction initialization."""
        content = "test content"
        action = SystemPromptReducerResetAction(content)
        assert action.content == content

    def test_reset_action_with_special_chars(self):
        """Test reset action with special characters."""
        content = "special!@#$%^&*()"
        action = SystemPromptReducerResetAction(content)
        result = system_prompt_reducer("old", action)
        assert result == content

    def test_multiple_sequential_appends(self):
        """Test multiple sequential append operations."""
        result = system_prompt_reducer("p1", "p2")
        result = system_prompt_reducer(result, "p3")
        result = system_prompt_reducer(result, "p4")

        assert "p1" in result
        assert "p2" in result
        assert "p3" in result
        assert "p4" in result
        assert result.count("\n\n") == 3

    def test_separator_is_double_newline(self):
        """Test that separator is specifically double newline."""
        result = system_prompt_reducer("a", "b")
        assert result == "a\n\nb"
        assert not result.startswith("\n\n")
        assert not result.endswith("\n\n")

    def test_reducer_preserves_content_exactly(self):
        """Test that reducer preserves content without modification."""
        special = "  spaced content  \twith\ttabs"
        result = system_prompt_reducer(special, "additional")
        assert special in result

    def test_reducer_with_unicode_content(self):
        """Test reducer with unicode characters."""
        result = system_prompt_reducer("首先", "然后")
        assert "首先" in result
        assert "然后" in result

    def test_reset_action_is_instance_check(self):
        """Test that isinstance check is used for reset action."""
        action = SystemPromptReducerResetAction("new")
        assert isinstance(action, SystemPromptReducerResetAction)

    def test_reset_action_different_from_string(self):
        """Test that reset action is different from string."""
        string_content = "new"
        action = SystemPromptReducerResetAction(string_content)

        result_string = system_prompt_reducer("old", string_content)
        result_action = system_prompt_reducer("old", action)

        # Different behavior
        assert result_string != result_action
        assert result_action == "new"
        assert result_string == "old\n\nnew"

    def test_very_long_prompts(self):
        """Test reducer with very long prompts."""
        long_left = "prompt " * 1000
        long_right = "additional " * 1000

        result = system_prompt_reducer(long_left, long_right)

        assert len(result) > len(long_left) + len(long_right)
        assert "prompt " in result
        assert "additional " in result

    def test_reducer_with_newlines_in_content(self):
        """Test reducer preserves existing newlines."""
        left = "line1\nline2"
        right = "line3\nline4"
        result = system_prompt_reducer(left, right)

        lines = result.split("\n")
        assert "line1" in lines
        assert "line2" in lines
        assert "line3" in lines
        assert "line4" in lines
