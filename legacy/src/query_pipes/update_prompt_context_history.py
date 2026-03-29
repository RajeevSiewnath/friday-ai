from pipelines.pipeline import Pipe
from models.query_context import QueryContext


class UpdatePromptContextHistory(Pipe[QueryContext]):

    def run(self, input):
        self.prompt_context.push_user(input.question)
        return input
