from pipelines.abstract_pipeline import AbstractPipe, PipeArg
from pipelines.query_pipeline import QueryContext


class QueryRewriter(AbstractPipe[QueryContext]):

    def rewrite_query(self, arg: PipeArg[QueryContext]):
        message = f"""You are in a conversation with a user, answering questions about {arg.prompt_context.user_context_short}.
You are about to look up information in a Knowledge Base to answer the user's question.

This is the history of your conversation so far with the user:
{arg.input.history}

And this is the user's current question:
{arg.input.question}

Respond only with a single, refined question that you will use to search the Knowledge Base.
It should be a VERY short specific question most likely to surface content. Focus on the question details.
Don't mention {arg.prompt_context.user} unless it's a general question about {arg.prompt_context.user}.
IMPORTANT: Respond ONLY with the knowledgebase query, nothing else.
"""
        return arg.llm.invoke(input=[{"role": "system", "content": message}])

    def pipe(self, arg):
        arg.input.question = self.rewrite_query(arg)
        return arg.input
