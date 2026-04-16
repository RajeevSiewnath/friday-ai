from friday.core.llm import LLM
from friday.query_nodes.nodes.capabilities_injector import capabilities_injector


def test_capabilities_injector_with_tools(make_runtime, llm: LLM):
    runtime = make_runtime({"llm": llm})
    state = {"system_prompt": ""}

    result = capabilities_injector(state, runtime)

    assert "system_prompt" in result
    assert "Capabilities:" in result["system_prompt"]
    assert "Send a message to Rajeev Siewnath" in result["system_prompt"]


def test_capabilities_injector_without_tools(make_runtime, llm: LLM):
    runtime = make_runtime({"llm": llm})
    state = {"system_prompt": ""}

    result = capabilities_injector(state, runtime)

    assert result == {}
