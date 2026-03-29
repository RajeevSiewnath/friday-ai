from core.llm import LLM
from core.prompt_context import PromptContext


def test_llm_init_with_default_model():
    """Test that LLM initializes with default model."""
    llm = LLM()
    assert llm.model == "gpt-4.1-nano"
    assert llm.embedding_model == "text-embedding-3-small"


def test_llm_init_with_custom_model():
    """Test that LLM initializes with custom model."""
    llm = LLM(model="gpt-4o", embedding_model="text-embedding-3-large")
    assert llm.model == "gpt-4o"
    assert llm.embedding_model == "text-embedding-3-large"


def test_llm_init_has_openai_client():
    """Test that LLM initializes with an OpenAI client."""
    llm = LLM()
    assert hasattr(llm, "client")
    assert llm.client is not None


def test_embedding_returns_vector(llm: LLM):
    """Test that embedding() returns a vector (list of floats)."""
    result = llm.embedding("test text")

    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(x, float) for x in result)


def test_embeddings_returns_multiple_vectors(llm: LLM):
    """Test that embeddings() returns multiple vectors."""
    texts = ["hello", "world", "test"]
    results = llm.embeddings(texts)

    assert isinstance(results, list)
    assert len(results) == 3
    assert all(isinstance(v, list) for v in results)
    assert all(all(isinstance(x, float) for x in v) for v in results)


def test_embeddings_vectors_have_consistent_dimension(llm: LLM):
    """Test that all embedding vectors have the same dimension."""
    texts = ["hello", "world", "test"]
    results = llm.embeddings(texts)

    dimensions = [len(v) for v in results]
    assert len(set(dimensions)) == 1  # All dimensions should be the same


def test_stream_returns_context_manager(llm: LLM, prompt_context: PromptContext):
    """Test that stream() returns a context manager for streaming responses."""
    stream = llm.stream(input=prompt_context.history)

    assert hasattr(stream, "__enter__")
    assert hasattr(stream, "__exit__")


def test_invoke_returns_string(llm: LLM, prompt_context: PromptContext):
    """Test that invoke() returns a string response."""
    prompt_context.push({"role": "user", "content": "Say hello"})

    result = llm.invoke(input=prompt_context.history)

    assert isinstance(result, str)
    assert len(result) > 0
