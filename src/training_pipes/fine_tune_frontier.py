import json
import tempfile
from openai.types.file_object import FileObject
from pipelines.abstract_pipeline import AbstractPipe
from models.training_program import (
    TrainingQuestion,
    TrainingProgram,
)


class FineTuneFrontier(AbstractPipe[TrainingProgram]):
    def __init__(
        self,
        suffix=None,
        model: str = None,
        n_epochs: int = 1,
        batch_size: int = 1,
        learning_rate_multiplier: float = 0.1,
    ):
        self.suffix = suffix
        self.model = model
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate_multiplier = learning_rate_multiplier

    def create_message(self, question: TrainingQuestion):
        self.prompt_context.reset()
        self.prompt_context.push(
            {"role": "user", "content": question.question},
            {"role": "assistant", "content": question.answer},
        )
        message_object = {"messages": self.prompt_context.history}
        return json.dumps(message_object)

    async def pipe(self, input):
        test_file: FileObject = None
        validation_file: FileObject = None

        with tempfile.NamedTemporaryFile(mode="w+", delete=True) as tmp_test:
            for question in input.training.questions:
                data = self.create_message(question)
                tmp_test.write(f"{data}\n")
                tmp_test.flush()

            with tempfile.NamedTemporaryFile(mode="w+", delete=True) as tmp_validation:
                for question in input.validation.questions:
                    data = self.create_message(question)
                    tmp_validation.write(f"{data}\n")
                    tmp_validation.flush()

                test_file = self.llm.upload_file(tmp_test.file, "fine-tune")
                validation_file = self.llm.upload_file(tmp_validation.file, "fine-tune")

        ft_job = self.llm.fine_tune(
            train_file_object=test_file,
            validate_file_object=validation_file,
            suffix=self.suffix,
            model=self.model,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            learning_rate_multiplier=self.learning_rate_multiplier,
        )

        await self.llm.wait_for_fine_tune(ft_job.id)
