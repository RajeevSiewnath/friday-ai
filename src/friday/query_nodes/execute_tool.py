from langgraph.runtime import Runtime
from friday.query_nodes.messages_state import MessagesState
from friday.query_nodes.query_context import QueryContext


async def execute_tool(state: MessagesState, runtime: Runtime[QueryContext]):
    tool_call = state["messages"][-1]
    result = runtime.context.llm.tool_shed.call(
        tool_call["name"], tool_call["arguments"]
    )
    return {
        "messages": [
            {
                "type": "function_call_output",
                "call_id": tool_call["call_id"],
                "output": str(result),
            }
        ]
    }
