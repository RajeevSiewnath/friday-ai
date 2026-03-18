from pipelines.pipeline import Pipe
from models.query_context import QueryContext


class QueryWriter(Pipe[str, QueryContext]):

    def run(self, input):
        return QueryContext(question=input)
