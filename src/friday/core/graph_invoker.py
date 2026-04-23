from copy import deepcopy
import json
from typing import AsyncGenerator, Callable, Generic, Optional, Type, get_type_hints
from langgraph.graph.state import CompiledStateGraph, ContextT, InputT, OutputT, StateT
from friday.loggers.logger import Logger

logger = Logger.get_logger("node.graph_invoker")


class GraphInvoker(Generic[StateT, ContextT, InputT, OutputT]):

    def __init__(
        self,
        graph: CompiledStateGraph[StateT, ContextT, InputT, OutputT],
        on_effect: Optional[Callable[[OutputT], None]] = None,
        on_node_change: Optional[Callable[[str], None]] = None,
        context: ContextT | None = None,
        update_state_with_effect: bool = True,
    ):
        self.graph: CompiledStateGraph[StateT, ContextT, InputT, OutputT] = graph
        self.on_effect: Optional[Callable[[OutputT], None]] = on_effect
        self.on_node_change: Optional[Callable[[str], None]] = on_node_change
        self.context = context
        self.update_state_with_effect: bool = update_state_with_effect

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
        reducers = self.__get_reducers(self.graph.builder.state_schema)
        s = deepcopy(state)
        logger.info("starting graph invocation...")
        logger.debug("initial state: %s", lambda: s)
        async for chunk in self.graph.astream(
            state,
            context=self.context,
            stream_mode=["custom", "updates", "values"],
            version="v2",
        ):
            if chunk["type"] == "values":
                s = deepcopy(chunk["data"])
                logger.debug("returning values: %s", lambda: s)
                yield s
            elif chunk["type"] == "updates":
                node_change = "-".join([item[0] for item in chunk["data"].items()])
                state_change = json.dumps([item[1] for item in chunk["data"].items()])
                logger.debug("entering node(s): %s", lambda: node_change)
                logger.debug("update state(s): %s", lambda: state_change)
                if self.on_node_change:
                    self.on_node_change(node_change)
            elif chunk["type"] == "custom":
                if self.on_effect:
                    self.on_effect(chunk["data"])
                s = self.__apply_reducers(s, chunk["data"], reducers)
                if self.update_state_with_effect:
                    logger.debug("yielding: %s", lambda: s)
                    yield s
        logger.info("graph invocation completed!")
        logger.debug("final state: %s", lambda: s)

    async def invoke(self, state: InputT) -> OutputT:
        s = state
        async for intermediate in self.stream(state):
            s = intermediate
        return s
