from pipelines.pipeline import Pipe
from models.query_context import QueryContext


class CapabilitiesInjector(Pipe[QueryContext]):
    def run(self, input):
        if len(self.llm.tool_shed.tools) > 0:
            self.prompt_context.capabilities = "\n\nCapabilities:\n" + "\n".join(
                f"- {tool.name}: {tool.definition["description"]}"
                for tool in self.llm.tool_shed.tools
            )
        else:
            self.prompt_context.capabilities = ""
        return input
