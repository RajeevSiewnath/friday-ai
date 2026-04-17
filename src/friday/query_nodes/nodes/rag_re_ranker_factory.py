from pydantic import BaseModel, Field
from langgraph.runtime import Runtime
from friday.query_nodes.contexts.llm_context import LLMContext
from friday.query_nodes.states.messages_state import MessagesState
from friday.query_nodes.states.rag_state import RagState
from friday.query_nodes.reducers.rag_reducer import RagReducerReplaceAction


class RankOrder(BaseModel):
    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )


class RagReRankerState(MessagesState, RagState):
    pass


def rag_re_ranker_factory(state_key: str):
    async def rag_re_ranker(state: RagReRankerState, runtime: Runtime[LLMContext]):
        system_prompt = """
You are a document re-ranker.
You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.
The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.
Reply only with the list of ranked chunk ids, nothing else. Include all the chunk ids you are provided with, reranked.
"""
        user_prompt = f"The user has asked the following question:\n\n{state["messages"][-1]["content"]}\n\nOrder all the chunks of text by relevance to the question, from most relevant to least relevant. Include all the chunk ids you are provided with, reranked.\n\n"
        user_prompt += "Here are the chunks:\n\n"
        for index, content in enumerate(
            query_output.document for query_output in state["rag_data"][state_key]
        ):
            user_prompt += f"# CHUNK ID: {index + 1}:\n\n{content}\n\n"
        user_prompt += "Reply only with the list of ranked chunk ids, nothing else."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = await runtime.context.llm.invoke(
            input=messages, response_format=RankOrder
        )
        return {
            "rag_data": {
                state_key: [
                    RagReducerReplaceAction(),
                    *[
                        state["rag_data"][state_key][i - 1]
                        for i in response.output_parsed.order
                    ],
                ]
            }
        }

    return rag_re_ranker
