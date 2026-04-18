import pytest
from friday.graph.query.reducers.stream_reducer import (
    stream_reducer,
    StreamReducerReplaceAction,
)


class TestStreamReducer:
    def test_reducer_appends_new_message(self):
        """Test reducer appends new messages."""
        left = [{"role": "user", "content": "hello"}]
        right = [{"role": "assistant", "content": "hi"}]

        result = stream_reducer(left, right)

        assert len(result) == 2
        assert result[1]["role"] == "assistant"

    def test_reducer_appends_content_to_existing_message(self):
        """Test reducer appends content to message with matching id."""
        left = [{"id": "msg1", "role": "user", "content": "hello"}]
        right = [{"id": "msg1", "content": " world"}]

        result = stream_reducer(left, right)

        assert len(result) == 1
        assert result[0]["content"] == "hello world"

    def test_reducer_empty_left(self):
        """Test reducer with empty left list."""
        left = []
        right = [{"role": "user", "content": "hello"}]

        result = stream_reducer(left, right)

        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_reducer_empty_right(self):
        """Test reducer with empty right list."""
        left = [{"role": "user", "content": "hello"}]
        right = []

        result = stream_reducer(left, right)

        assert len(result) == 1
        assert result[0]["content"] == "hello"

    def test_reducer_both_empty(self):
        """Test reducer with both empty lists."""
        left = []
        right = []

        result = stream_reducer(left, right)

        assert result == []

    def test_reducer_multiple_messages(self):
        """Test reducer with multiple messages."""
        left = [
            {"id": "msg1", "role": "user", "content": "hello"},
            {"id": "msg2", "role": "assistant", "content": "hi"},
        ]
        right = [
            {"id": "msg1", "content": " there"},
            {"id": "msg3", "role": "user", "content": "new"},
        ]

        result = stream_reducer(left, right)

        assert len(result) == 3
        assert result[0]["content"] == "hello there"
        assert result[1]["content"] == "hi"
        assert result[2]["content"] == "new"

    def test_reducer_with_replace_action(self):
        """Test reducer with StreamReducerReplaceAction."""
        left = [{"id": "msg1", "role": "user", "content": "old"}]
        action = StreamReducerReplaceAction("new")
        right = [{"id": "msg1", "content": action}]

        result = stream_reducer(left, right)

        assert len(result) == 1
        assert result[0]["content"] == "new"

    def test_reducer_updates_non_content_fields(self):
        """Test reducer updates non-content fields."""
        left = [{"id": "msg1", "role": "user", "content": "hello", "status": "pending"}]
        right = [{"id": "msg1", "status": "completed"}]

        result = stream_reducer(left, right)

        assert result[0]["status"] == "completed"
        assert result[0]["content"] == "hello"

    def test_reducer_system_message_appending(self):
        """Test reducer appends to system message."""
        left = [{"role": "system", "content": "system1"}]
        right = [{"role": "system", "content": "system2"}]

        result = stream_reducer(left, right)

        assert len(result) == 1
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "system1system2"

    def test_reducer_without_message_id(self):
        """Test reducer with messages without id."""
        left = [{"role": "user", "content": "hello"}]
        right = [{"role": "user", "content": " there"}]

        result = stream_reducer(left, right)

        # Should append as new message since no id match
        assert len(result) == 2

    def test_reducer_preserves_message_order(self):
        """Test reducer preserves message order."""
        left = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
            {"role": "user", "content": "third"},
        ]
        right = [{"role": "assistant", "content": " response"}]

        result = stream_reducer(left, right)

        assert result[0]["content"] == "first"
        assert result[1]["content"] == "second"

    def test_reducer_deepcopy_isolation(self):
        """Test reducer doesn't modify original left."""
        left = [{"id": "msg1", "role": "user", "content": "hello"}]
        original_content = left[0]["content"]
        right = [{"id": "msg1", "content": " modified"}]

        result = stream_reducer(left, right)

        # Original left should be unchanged
        assert left[0]["content"] == original_content

    def test_replace_action_initialization(self):
        """Test StreamReducerReplaceAction initialization."""
        content = "replacement"
        action = StreamReducerReplaceAction(content)
        assert action.content == content

    def test_replace_action_with_empty_string(self):
        """Test replace action with empty string."""
        action = StreamReducerReplaceAction("")
        result = stream_reducer(
            [{"id": "msg1", "content": "old"}],
            [{"id": "msg1", "content": action}]
        )
        assert result[0]["content"] == ""

    def test_replace_action_with_multiline(self):
        """Test replace action with multiline content."""
        multiline = "line1\nline2\nline3"
        action = StreamReducerReplaceAction(multiline)
        result = stream_reducer(
            [{"id": "msg1", "content": "old"}],
            [{"id": "msg1", "content": action}]
        )
        assert result[0]["content"] == multiline

    def test_reducer_message_with_metadata(self):
        """Test reducer preserves metadata."""
        left = [
            {
                "id": "msg1",
                "role": "user",
                "content": "hello",
                "timestamp": "2024-01-01",
            }
        ]
        right = [{"id": "msg1", "content": " world"}]

        result = stream_reducer(left, right)

        assert result[0]["timestamp"] == "2024-01-01"
        assert result[0]["content"] == "hello world"

    def test_reducer_multiple_appends_same_message(self):
        """Test multiple appends to same message."""
        left = [{"id": "msg1", "role": "user", "content": "Hello"}]

        right1 = [{"id": "msg1", "content": " world"}]
        result = stream_reducer(left, right1)
        assert result[0]["content"] == "Hello world"

        right2 = [{"id": "msg1", "content": "!"}]
        result = stream_reducer(result, right2)
        assert result[0]["content"] == "Hello world!"

    def test_reducer_with_special_characters(self):
        """Test reducer with special characters."""
        left = [{"id": "msg1", "content": "hello@#$"}]
        right = [{"id": "msg1", "content": "%^&"}]

        result = stream_reducer(left, right)

        assert "hello@#$%^&" == result[0]["content"]

    def test_reducer_with_unicode(self):
        """Test reducer with unicode content."""
        left = [{"id": "msg1", "content": "你好"}]
        right = [{"id": "msg1", "content": "世界"}]

        result = stream_reducer(left, right)

        assert result[0]["content"] == "你好世界"

    def test_reducer_large_message_list(self):
        """Test reducer with many messages."""
        left = [{"id": f"msg{i}", "role": "user", "content": f"msg{i}"} for i in range(100)]
        right = [{"id": "msg50", "content": " updated"}]

        result = stream_reducer(left, right)

        assert len(result) == 100
        assert result[50]["content"] == "msg50 updated"

    def test_reducer_append_without_content_field(self):
        """Test reducer behavior when message has no content field."""
        left = [{"id": "msg1", "role": "user", "content": "hello"}]
        right = [{"id": "msg1", "status": "done"}]

        result = stream_reducer(left, right)

        assert result[0]["content"] == "hello"
        assert result[0]["status"] == "done"

    def test_reducer_right_without_content_field(self):
        """Test reducer when right entry has no content field."""
        left = [{"id": "msg1", "role": "user", "content": "hello"}]
        right = [{"id": "msg1", "role": "user"}]

        result = stream_reducer(left, right)

        assert result[0]["content"] == "hello"
        assert result[0]["role"] == "user"

    def test_reduce_action_different_from_string(self):
        """Test that replace action behaves differently from string."""
        string_content = "new"
        action = StreamReducerReplaceAction(string_content)

        result_string = stream_reducer(
            [{"id": "msg1", "content": "old"}],
            [{"id": "msg1", "content": string_content}]
        )
        result_action = stream_reducer(
            [{"id": "msg1", "content": "old"}],
            [{"id": "msg1", "content": action}]
        )

        # Different behavior
        assert result_string[0]["content"] != result_action[0]["content"]
        assert result_action[0]["content"] == "new"
        assert result_string[0]["content"] == "oldnew"
