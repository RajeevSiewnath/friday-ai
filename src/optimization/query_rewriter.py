from typing import Any

from invocation.llm import invoke_llm
from optimization.Chunk import ChunkResult


SYSTEM_PROMPT = """
You are a personal job agent for Rajeev Siewnath. 
You provide information about his curriculum vitae to the user.
Your answer will be evaluated for accuracy, relevance and completeness, so make sure it only answers the question and fully answers it.
If you don't know the answer, say so.
For context, here are specific extracts from the Knowledge Base that might be directly relevant to the user's question:
{context}

With this context, please answer the user's question. Be accurate, relevant and complete.
"""


def make_rag_messages(question: str, history: list[Any], chunks: list[ChunkResult]):
    """Make RAG messages"""
    context = "\n\n".join(
        f"Extract from {chunk.metadata['source']}:\n{chunk.page_content}"
        for chunk in chunks
    )
    system_prompt = SYSTEM_PROMPT.format(context=context)
    return (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": question}]
    )


def rewrite_query(question: str, history: list[Any] = []):
    """Rewrite the user's question to be a more specific question that is more likely to surface relevant content in the Knowledge Base."""
    message = f"""
You are in a conversation with a user, answering questions about Rajeev Siewnath's curriculum vitae.
You are about to look up information in a Knowledge Base to answer the user's question.

This is the history of your conversation so far with the user:
{history}

And this is the user's current question:
{question}

Respond only with a single, refined question that you will use to search the Knowledge Base.
It should be a VERY short specific question most likely to surface content. Focus on the question details.
Don't mention ENSER uNMENTIONED unless it's a general question about the company.
IMPORTANT: Respond ONLY with the knowledgebase query, nothing else.
"""
    return invoke_llm(input=[{"role": "system", "content": message}])


# def answer_question(question: str, history: list[dict] = []) -> tuple[str, list]:
#     """
#     Answer a question using RAG and return the answer and the retrieved context
#     """
#     query = rewrite_query(question, history)
#     chunks = re_rank(query)
#     messages = make_rag_messages(question, history, chunks)
#     response = completion(model=MODEL, messages=messages)
#     return response.choices[0].message.content, chunks