from core.prompt_context import PromptContext


def test_system_message_formatting():
    """Test that system_message formats the template with available variables."""
    context = PromptContext(
        message="User: {user}\nContext: {context}", user="Alice", context="Test context"
    )

    assert context.system_message == "User: Alice\nContext: Test context"


def test_history_includes_system_message(prompt_context):
    """Test that history includes the system message as the first item."""
    assert len(prompt_context.history) >= 1
    assert prompt_context.history[0]["role"] == "system"
    assert prompt_context.history[0]["content"] == prompt_context.system_message


def test_history_extends_with_entries(prompt_context):
    """Test that history includes both system message and added entries."""
    prompt_context.push({"role": "user", "content": "Hello"})
    prompt_context.push({"role": "assistant", "content": "Hi"})

    assert len(prompt_context.history) == 3
    assert prompt_context.history[1]["role"] == "user"
    assert prompt_context.history[2]["role"] == "assistant"


def test_conversation_filters_by_role(prompt_context):
    """Test that conversation property only includes user and assistant messages."""
    prompt_context.push({"role": "user", "content": "Hello"})
    prompt_context.push({"role": "assistant", "content": "Hi"})
    prompt_context.push({"role": "system", "content": "System"})

    conversation = prompt_context.conversation
    roles = [entry.get("role") for entry in conversation]

    assert "system" not in roles
    assert "user" in roles
    assert "assistant" in roles


def test_push_adds_to_history(prompt_context):
    """Test that push() adds entries to internal history."""
    entry = {"role": "user", "content": "Test"}
    prompt_context.push(entry)

    assert entry in prompt_context._history


def test_push_returns_self(prompt_context):
    """Test that push() returns self for method chaining."""
    result = prompt_context.push({"role": "user", "content": "Test"})
    assert result is prompt_context


def test_reset_clears_history(prompt_context):
    """Test that reset() clears the internal history."""
    prompt_context.push({"role": "user", "content": "Message 1"})
    prompt_context.push({"role": "assistant", "content": "Message 2"})

    assert len(prompt_context._history) == 2

    prompt_context.reset()

    assert len(prompt_context._history) == 0


def test_reset_returns_self(prompt_context):
    """Test that reset() returns self for method chaining."""
    result = prompt_context.reset()
    assert result is prompt_context
