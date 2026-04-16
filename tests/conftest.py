import pytest
from friday.core.llm import LLM
from friday.core.tool_shed import ToolShed
from friday.core.vector_db import VectorDB


@pytest.fixture
def llm():
    def send_contact_request(message: str) -> bool:
        """
        Send a message to Rajeev Siewnath.

        Args:
            message: The message to send

        Returns:
            Whether the message was sent successfully
        """
        # print("Sending:", message)
        return True

    tool_shed = ToolShed(send_contact_request)
    return LLM(tool_shed=tool_shed)


@pytest.fixture
def vector_db():
    return VectorDB()
