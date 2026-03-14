from pipelines.abstract_pipeline import AbstractPipe
from pipelines.query_pipeline import QueryContext


class QueryRewriter(AbstractPipe[QueryContext]):

    def rewrite_query(self, input: QueryContext):
        message = f"""You are in a conversation with a user, answering questions about {self.prompt_context.user_context_short}.
You are about to look up information in a Knowledge Base to answer the user's question.

This is the history of your conversation so far with the user:
{self.prompt_context.history[1:]}

And this is the user's current question:
{input.question}

Respond only with a single, refined question that you will use to search the Knowledge Base.
It should be a VERY short specific question most likely to surface content. Focus on the question details.
Don't mention {self.prompt_context.user} unless it's a general question about {self.prompt_context.user}.
IMPORTANT: Respond ONLY with the knowledgebase query, nothing else.
"""
        return self.llm.invoke(input=[{"role": "system", "content": message}])

    def pipe(self, input):
        input.question = self.rewrite_query(input)
        return input
