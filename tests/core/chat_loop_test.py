from core.chat_loop import ChatLoop
from core.llm import LLM
from core.prompt_context import PromptContext


def test_invoke_with_message(llm: LLM, prompt_context: PromptContext):
    """Test that invoke() processes streaming events and updates history."""
    chat_loop = ChatLoop(llm, prompt_context)

    # Submit a message
    prompt_context.push({"role": "user", "content": "Hello, what's on my CV?"})

    # History should include system message + user message
    assert len(prompt_context.history) == 2
    assert prompt_context.history[0]["role"] == "system"
    assert prompt_context.history[1]["role"] == "user"
    assert prompt_context.history[1]["content"] == "Hello, what's on my CV?"

    # Invoke and collect results
    results = list(chat_loop.invoke())

    # Should have yielded at least one value
    assert len(results) > 0

    # Should have True at the end (response completed or error occurred)
    assert True in results

    # History should be updated with assistant response
    assert len(prompt_context.history) > 2
    assert prompt_context.history[-1]["role"] == "assistant"
