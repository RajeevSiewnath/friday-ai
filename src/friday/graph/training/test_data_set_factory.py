from rapidfuzz import fuzz
from langgraph.runtime import Runtime
from friday.graph.query.contexts.llm_context import LLMContext
from friday.graph.training.states.statistics_state import StatisticsState
from friday.graph.training.states.training_state import TrainingState
from friday.loggers.logger import Logger


class TestDataSetState(TrainingState, StatisticsState):
    pass


def test_data_set_factory(statistics_key: str):
    async def test_data_set(state: TestDataSetState, runtime: Runtime[LLMContext]):
        logger = Logger.get_logger("node.test_data_set")
        logger.debug("testing data set")

        fuzziness = 0
        for question in state["testing_data_set"]:
            logger.trace("question: %s", lambda: question)
            response = await runtime.context.llm.invoke(
                [{"role": "user", "content": question.question}]
            )
            logger.trace("response: %s", lambda: response)
            fuzziness += fuzz.ratio(response.output_text, question.answer) / 100
        return {
            "statistics": {statistics_key: fuzziness / len(state["testing_data_set"])}
        }

    return test_data_set
