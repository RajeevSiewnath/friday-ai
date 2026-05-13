from typing import Any, TypedDict
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from friday.core.llm import LLM
from friday.graph.query.contexts.llm_context import LLMContext
from friday.graph.query.reducers.stream_reducer import StreamReducerClearAction
from friday.graph.query.states.messages_state import MessagesState
from friday.loggers.logger import Logger


class FunctionBaseState(TypedDict):
    id: str


class FunctionDefinitionState(FunctionBaseState):
    name: str
    args: dict


class FunctionResultState(FunctionBaseState):
    results: Any


def tool_call_graph(llm: LLM) -> StateGraph:
    def get(
        state: MessagesState, runtime: Runtime[LLMContext]
    ) -> FunctionDefinitionState:
        logger = Logger.get_logger("node.tool_call_graph.get")
        tool_call = state["messages"][-1]
        logger.debug(
            'searching for function for call id "%s"...', lambda: tool_call["call_id"]
        )
        logger.trace(
            "functions: %s",
            lambda: [
                t
                for t in runtime.context.llm.tool_shed.tools
                if t.name == tool_call["name"]
            ],
        )

        tool = next(
            (
                t
                for t in runtime.context.llm.tool_shed.tools
                if t.name == tool_call["name"]
            ),
            None,
        )
        if tool:
            return {
                "id": tool_call["call_id"],
                "name": tool_call["name"],
                "args": tool_call["arguments"],
            }
        else:
            raise f"tool not defined: '{tool_call["name"]}'"

    def call_factory(name: str):
        async def call(
            state: FunctionDefinitionState, runtime: Runtime[LLMContext]
        ) -> FunctionResultState:
            logger = Logger.get_logger("node.tool_call_graph.call")
            logger.debug(
                'calling "%s" for call id "%s"...',
                lambda: name,
                lambda: state["id"],
            )
            return {
                "results": str(
                    await runtime.context.llm.tool_shed.call(
                        state["name"], state["args"]
                    )
                )
            }

        return call

    def _return(state: FunctionResultState) -> MessagesState:
        logger = Logger.get_logger("node.tool_call_graph.return")
        logger.debug('returning function value for call id "%s"', lambda: state["id"])
        logger.trace("return value: %s", state["results"])

        return {
            "messages": [
                StreamReducerClearAction(
                    {
                        "type": "function_call_output",
                        "call_id": state["id"],
                        "output": state["results"],
                    }
                )
            ]
        }

    graph = StateGraph(MessagesState)
    graph.add_node("get", get)
    edges = []
    for t in llm.tool_shed.tools:
        graph.add_node(t.name, call_factory(t.name))
        edges.append(t.name)
    graph.add_node("return", _return)

    graph.add_edge(START, "get")
    graph.add_conditional_edges(
        "get", lambda state: state["name"], {edge: edge for edge in edges}
    )
    for edge in edges:
        graph.add_edge(edge, "return")
    graph.add_edge("return", END)

    return graph
