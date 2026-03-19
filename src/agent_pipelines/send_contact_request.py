from pipelines.pipeline import Pipeline


class SendContactRequest(Pipeline[str, bool]):
    def run(self, input):
        """
        Use this tool when the user wants to send a message or contact Rajeev Siewnath.

        Args:
          input: The message the user wants to send to Rajeev Siewnath

        Returns:
          A boolean whether the message was sent successfully or not
        """
        return super().run(input)
