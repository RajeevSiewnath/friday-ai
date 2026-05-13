from copy import deepcopy
from typing import AsyncGenerator, Callable, Generic, Optional, Type, get_type_hints
from langgraph.graph.state import CompiledStateGraph, ContextT, InputT, OutputT, StateT
from friday.loggers.logger import Logger
from langgraph.types import CustomStreamPart, DebugStreamPart, UpdatesStreamPart

logger = Logger.get_logger("node.graph_invoker")


class GraphInvoker(Generic[StateT, ContextT, InputT, OutputT]):

    def __init__(
        self,
        graph: CompiledStateGraph[StateT, ContextT, InputT, OutputT],
        on_node_change: Optional[Callable[[UpdatesStreamPart], None]] = None,
        on_effect: Optional[Callable[[CustomStreamPart], None]] = None,
        on_debug: Optional[Callable[[DebugStreamPart], None]] = None,
        on_nodes: Optional[Callable[[list[str], list[str]], None]] = None,
        context: ContextT | None = None,
        yield_on_state_change: bool = True,
        yield_on_node_change: bool = False,
        yield_on_debug: bool = False,
    ):
        self.graph: CompiledStateGraph[StateT, ContextT, InputT, OutputT] = graph
        self.on_node_change: Optional[Callable[[UpdatesStreamPart], None]] = (
            on_node_change
        )
        self.on_effect: Optional[Callable[[CustomStreamPart], None]] = on_effect
        self.on_debug: Optional[Callable[[DebugStreamPart], None]] = on_debug
        self.on_nodes: Optional[Callable[[list[str], list[str]], None]] = on_nodes
        self.context = context
        self.yield_on_state_change: bool = yield_on_state_change
        self.yield_on_node_change: bool = yield_on_node_change
        self.yield_on_debug: bool = yield_on_debug

    def __get_reducers(self, state_type: Type[StateT]):
        reducers = {}

        hints = get_type_hints(state_type, include_extras=True)

        for key, hint in hints.items():
            metadata = getattr(hint, "__metadata__", None)
            if metadata:
                reducers[key] = metadata[0]

        return reducers

    def __apply_reducers(self, state: dict, update: dict, reducers: dict):
        new_state = dict(state)

        for key, value in update.items():
            if key in reducers:
                old_value = state.get(key)
                new_state[key] = reducers[key](old_value, value)
            else:
                new_state[key] = value

        return new_state

    async def stream(self, state: InputT) -> AsyncGenerator[OutputT, None]:
        opened_nodes = []
        closed_nodes = []
        reducers = self.__get_reducers(self.graph.builder.state_schema)
        s = deepcopy(state)
        logger.info("starting graph invocation...")
        logger.trace("initial state: %s", lambda: s)
        async for chunk in self.graph.astream(
            state,
            context=self.context,
            stream_mode=["custom", "updates", "values", "debug"],
            version="v2",
            subgraphs=True,
        ):
            if chunk["type"] == "values":
                s = deepcopy(chunk["data"])
                logger.trace("returning values: %s", lambda: s)
                yield s
            elif chunk["type"] == "updates":
                logger.trace(
                    "exiting node(s): %s",
                    lambda: ",".join([item[0] for item in chunk["data"].items()]),
                )
                logger.trace(
                    "update state(s): %s",
                    lambda: [item[1] for item in chunk["data"].items()],
                )
                if self.on_node_change:
                    self.on_node_change(chunk)

                if self.yield_on_node_change:
                    yield s
            elif chunk["type"] == "custom":
                if self.on_effect:
                    self.on_effect(chunk)
                s = self.__apply_reducers(s, chunk["data"], reducers)
                if self.yield_on_state_change:
                    logger.trace("yielding: %s", lambda: s)
                    yield s
            elif chunk["type"] == "debug":
                ns = [n.split(":")[0] for n in list(chunk["ns"])]
                name = ":".join([*ns, chunk["data"]["payload"]["name"]])
                if chunk["data"]["type"] == "task":
                    closed_nodes = []
                    opened_nodes.append(name)
                    logger.info(f"entering node {name}...")
                elif chunk["data"]["type"] == "task_result":
                    opened_nodes.remove(name)
                    closed_nodes.append(name)
                    logger.info(f"exiting node {name}...")

                if self.on_nodes:
                    self.on_nodes(opened_nodes, closed_nodes)

                if self.on_debug:
                    self.on_debug(chunk)

                if self.yield_on_node_change:
                    yield s

        logger.info("graph invocation completed!")
        logger.trace("final state: %s", lambda: s)

    async def invoke(self, state: InputT) -> OutputT:
        s = state
        async for intermediate in self.stream(state):
            s = intermediate
        return s
