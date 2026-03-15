from pipelines.abstract_pipeline import AbstractPipe
from models.training_program import (
    TrainingProgram,
)
from rapidfuzz import fuzz


class FineTuneFrontier(AbstractPipe[TrainingProgram]):
    def pipe(self, input):
        fuzziness = 0
        for question in input.testing.questions:
            self.prompt_context.reset()
            self.prompt_context.push({"role": "user", "content": question.question})
            response = self.llm.invoke(self.prompt_context.history)
            fuzziness += fuzz.ratio(response, question.answer) / 100
        input.statistics["test"] = fuzziness / len(input.testing.questions)
