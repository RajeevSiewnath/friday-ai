from copy import deepcopy
from typing import AsyncGenerator, Callable, Generic, Optional, Type, get_type_hints
from langgraph.graph.state import CompiledStateGraph, ContextT, InputT, OutputT, StateT


class GraphInvoker(Generic[StateT, ContextT, InputT, OutputT]):

    def __init__(
        self,
        graph: CompiledStateGraph[StateT, ContextT, InputT, OutputT],
        on_effect: Optional[Callable[[OutputT], None]] = None,
        on_node_change: Optional[Callable[[str], None]] = None,
        update_state_with_effect: bool = True,
    ):
        self.graph: CompiledStateGraph[StateT, ContextT, InputT, OutputT] = graph
        self.on_effect: Optional[Callable[[OutputT], None]] = on_effect
        self.on_node_change: Optional[Callable[[str], None]] = on_node_change
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

    async def stream(
        self,
        state: InputT,
    ) -> AsyncGenerator[OutputT, None]:
        reducers = self.__get_reducers(self.graph.builder.state_schema)
        s = deepcopy(state)
        async for chunk in self.graph.astream(
            state, stream_mode=["custom", "updates", "values"], version="v2"
        ):
            if chunk["type"] == "values":
                s = deepcopy(state)
                yield chunk["data"]
            elif chunk["type"] == "updates":
                if self.on_node_change:
                    self.on_node_change(
                        "-".join([item[0] for item in chunk["data"].items()])
                    )
            elif chunk["type"] == "custom":
                if self.on_effect:
                    self.on_effect(chunk["data"])
                s = self.__apply_reducers(s, chunk["data"], reducers)
                if self.update_state_with_effect:
                    yield s

    async def invoke(self, state: InputT) -> OutputT:
        s = state
        async for intermediate in self.stream(state):
            s = intermediate
        return s
