import pytest
from typing import get_type_hints, get_origin, get_args
from friday.query_nodes.states.messages_state import MessagesState
from friday.query_nodes.reducers.stream_reducer import (
    stream_reducer,
    StreamReducerReplaceAction,
)


class TestMessagesState:
    def test_state_has_messages_field(self):
        """Test that MessagesState has the messages field."""
        assert "messages" in MessagesState.__annotations__

    def test_messages_field_is_list(self):
        """Test that messages field is annotated as list."""
        annotation = MessagesState.__annotations__["messages"]
        origin = get_origin(annotation)
        # Should be Annotated type
        assert origin is not None

    def test_state_is_typed_dict(self):
        """Test that MessagesState is a TypedDict."""
        assert hasattr(MessagesState, "__annotations__")
        assert isinstance(MessagesState.__annotations__, dict)

    def test_stream_reducer_appends_new_message(self):
        """Test that stream_reducer appends new messages."""
        left = [{"role": "user", "content": "hello"}]
        right = [{"role": "assistant", "content": "hi"}]

        result = stream_reducer(left, right)

        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"

    def test_stream_reducer_appends_content_to_existing_message(self):
        """Test that stream_reducer appends content to message with same id."""
        left = [{"id": "msg1", "role": "user", "content": "hello"}]
        right = [{"id": "msg1", "content": " world"}]

        result = stream_reducer(left, right)

        assert len(result) == 1
        assert result[0]["content"] == "hello world"

    def test_stream_reducer_with_replace_action(self):
        """Test stream_reducer with StreamReducerReplaceAction."""
        left = [{"id": "msg1", "role": "user", "content": "old"}]
        action = StreamReducerReplaceAction("new")
        right = [{"id": "msg1", "content": action}]

        result = stream_reducer(left, right)

        assert len(result) == 1
        assert result[0]["content"] == "new"

    def test_stream_reducer_updates_non_content_fields(self):
        """Test that stream_reducer updates non-content fields."""
        left = [{"id": "msg1", "role": "user", "content": "hello", "status": "pending"}]
        right = [{"id": "msg1", "status": "completed"}]

        result = stream_reducer(left, right)

        assert result[0]["status"] == "completed"
        assert result[0]["content"] == "hello"

    def test_stream_reducer_multiple_messages(self):
        """Test stream_reducer with multiple messages."""
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

    def test_stream_reducer_system_message_handling(self):
        """Test stream_reducer handles system messages specially."""
        left = [{"role": "system", "content": "system1"}]
        right = [{"role": "system", "content": "system2"}]

        result = stream_reducer(left, right)

        assert len(result) == 1
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "system1system2"

    def test_stream_reducer_system_message_handling_when_present(self):
        """Test stream_reducer when system message already exists in left."""
        left = [
            {"role": "system", "content": "initial system"},
            {"role": "user", "content": "hello"},
        ]
        right = [{"role": "system", "content": " prompt"}]

        result = stream_reducer(left, right)

        # System message should be updated with appended content
        system_msg = next((m for m in result if m.get("role") == "system"), None)
        assert system_msg is not None
        assert "initial system" in system_msg["content"]
        assert "prompt" in system_msg["content"]

    def test_stream_reducer_preserves_message_order(self):
        """Test that stream_reducer preserves message order."""
        left = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
            {"role": "user", "content": "third"},
        ]
        right = [{"role": "assistant", "content": " response"}]

        result = stream_reducer(left, right)

        # New assistant message should be appended
        assert len(result) == 4
        assert result[0]["content"] == "first"

    def test_stream_reducer_deepcopy_isolation(self):
        """Test that stream_reducer doesn't modify original left."""
        left = [{"id": "msg1", "role": "user", "content": "hello"}]
        original_left = [{"id": "msg1", "role": "user", "content": "hello"}]
        right = [{"id": "msg1", "content": " modified"}]

        result = stream_reducer(left, right)

        # Original left should be unchanged
        assert left[0]["content"] == "hello"

    def test_stream_reducer_empty_left(self):
        """Test stream_reducer with empty left."""
        left = []
        right = [{"role": "user", "content": "hello"}]

        result = stream_reducer(left, right)

        assert len(result) == 1
        assert result[0]["content"] == "hello"

    def test_stream_reducer_empty_right(self):
        """Test stream_reducer with empty right."""
        left = [{"role": "user", "content": "hello"}]
        right = []

        result = stream_reducer(left, right)

        assert len(result) == 1
        assert result[0]["content"] == "hello"

    def test_stream_reducer_message_without_id(self):
        """Test stream_reducer handles messages without id."""
        left = [{"role": "user", "content": "hello"}]
        right = [{"role": "user", "content": " there"}]

        result = stream_reducer(left, right)

        # Should append as new message since no id match
        assert len(result) == 2

    def test_stream_replace_action_initialization(self):
        """Test StreamReducerReplaceAction initialization."""
        content = "replacement text"
        action = StreamReducerReplaceAction(content)
        assert action.content == content

    def test_state_instance_creation(self):
        """Test creating MessagesState instance."""
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        state = MessagesState(messages=messages)
        assert state["messages"] == messages
        assert len(state["messages"]) == 2

    def test_state_with_empty_messages(self):
        """Test MessagesState with empty messages."""
        state = MessagesState(messages=[])
        assert state["messages"] == []

    def test_stream_reducer_message_with_metadata(self):
        """Test stream_reducer preserves message metadata."""
        left = [
            {"id": "msg1", "role": "user", "content": "hello", "timestamp": "2024-01-01"}
        ]
        right = [{"id": "msg1", "content": " world"}]

        result = stream_reducer(left, right)

        assert result[0]["timestamp"] == "2024-01-01"
        assert result[0]["content"] == "hello world"

    def test_stream_reducer_multiple_appends_to_same_message(self):
        """Test multiple appends to same message."""
        left = [{"id": "msg1", "role": "assistant", "content": "Hello"}]
        right1 = [{"id": "msg1", "content": " world"}]

        result = stream_reducer(left, right1)
        assert result[0]["content"] == "Hello world"

        right2 = [{"id": "msg1", "content": "!"}]
        result = stream_reducer(result, right2)
        assert result[0]["content"] == "Hello world!"
