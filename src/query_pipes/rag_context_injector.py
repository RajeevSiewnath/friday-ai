from pipelines.pipeline import Pipe
from models.query_context import QueryContext


class RagContextInjector(Pipe[QueryContext]):
    def run(self, input):
        self.prompt_context.context = "\n\nContext:\n" + "\n\n".join(
            context.content for context in input.context.contexts
        )
        return input
