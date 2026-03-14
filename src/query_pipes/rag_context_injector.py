from pipelines.abstract_pipeline import AbstractPipe
from models.query_context import QueryContext


class RagContextInjector(AbstractPipe[QueryContext]):
    def pipe(self, input):
        self.prompt_context.context = "\n\n".join(
            context.content for context in input.context.contexts
        )
        return input
