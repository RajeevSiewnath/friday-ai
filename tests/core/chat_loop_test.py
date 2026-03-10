from core.chat_loop import ChatLoop, Role


def test_invoke_with_message(llm, prompt_context):
    """Test that invoke() processes streaming events and updates history."""
    chat_loop = ChatLoop(llm, prompt_context, tools=[])

    # Submit a message
    chat_loop.submit_message("Hello, what's on my CV?")

    # History should include system message + user message
    assert len(prompt_context.history) == 2
    assert prompt_context.history[0]["role"] == "system"
    assert prompt_context.history[1]["role"] == Role.USER
    assert prompt_context.history[1]["content"] == "Hello, what's on my CV?"

    # Invoke and collect results
    results = list(chat_loop.invoke())

    # Should have yielded at least one value
    assert len(results) > 0

    # Should have True at the end (response completed or error occurred)
    assert True in results

    # History should be updated with assistant response
    assert len(prompt_context.history) > 2
    assert prompt_context.history[-1]["role"] == Role.ASSISTANT


def test_reset_clears_history(llm, prompt_context):
    """Test that reset() clears the conversation history."""
    chat_loop = ChatLoop(llm, prompt_context, tools=[])

    # Submit a message
    chat_loop.submit_message("Test message")
    assert len(prompt_context._history) == 1

    # Reset should clear history
    chat_loop.reset()
    assert len(prompt_context._history) == 0
