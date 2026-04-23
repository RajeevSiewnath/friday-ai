import json
import tempfile
from langgraph.runtime import Runtime
from openai.types.file_object import FileObject
from friday.core.evaluation import Evaluation
from friday.graph.query.contexts.llm_context import LLMContext
from friday.graph.training.states.training_state import TrainingState
from friday.loggers.logger import Logger


def fine_tune_frontier_factory(
    suffix: str = None,
    model: str = None,
    n_epochs: int = 1,
    batch_size: int = 1,
    learning_rate_multiplier: float = 0.1,
    wait_for_fine_tune_to_complete: bool = False,
):

    def create_message(question: Evaluation):
        messages = [
            {
                {"role": "user", "content": question.question},
                {"role": "assistant", "content": question.answer},
            }
        ]
        message_object = {"messages": messages}
        return json.dumps(message_object)

    async def fine_tune_frontier(state: TrainingState, runtime: Runtime[LLMContext]):
        logger = Logger.get_logger("node.fine_tune_frontier")
        logger.info("fine tuning frontier model")

        test_file: FileObject = None
        validation_file: FileObject = None

        with tempfile.NamedTemporaryFile(mode="w+", delete=True) as tmp_test:
            for question in state["training_data_set"]:
                data = create_message(question)
                tmp_test.write(f"{data}\n")
                tmp_test.flush()

            with tempfile.NamedTemporaryFile(mode="w+", delete=True) as tmp_validation:
                for question in state["validating_data_set"]:
                    data = create_message(question)
                    tmp_validation.write(f"{data}\n")
                    tmp_validation.flush()

                test_file = await runtime.context.llm.upload_file(
                    tmp_test.file, "fine-tune"
                )
                validation_file = await runtime.context.llm.upload_file(
                    tmp_validation.file, "fine-tune"
                )

        logger.debug("test file: %s", lambda: test_file)
        logger.debug("validation file: %s", lambda: validation_file)

        ft_job = await runtime.context.llm.fine_tune(
            train_file_object=test_file,
            validate_file_object=validation_file,
            suffix=suffix,
            model=model,
            n_epochs=n_epochs,
            batch_size=batch_size,
            learning_rate_multiplier=learning_rate_multiplier,
        )
        logger.debug("fine-tune job: %s", lambda: ft_job)

        if wait_for_fine_tune_to_complete:
            await runtime.context.llm.wait_for_fine_tune(ft_job.id)

        return fine_tune_frontier
