from rapidfuzz import fuzz
from langgraph.runtime import Runtime
from friday.graph.query.contexts.llm_context import LLMContext
from friday.graph.training.states.statistics_state import StatisticsState
from friday.graph.training.states.training_state import TrainingState


class TestDataSetState(TrainingState, StatisticsState):
    pass


def test_data_set_factory(statistics_key: str):
    async def test_data_set(state: TestDataSetState, runtime: Runtime[LLMContext]):
        fuzziness = 0
        for question in state["testing_data_set"]:
            response = await runtime.context.llm.invoke(
                [{"role": "user", "content": question.question}]
            )
            fuzziness += fuzz.ratio(response, question.answer) / 100
        return {
            "statistics": {statistics_key: fuzziness / len(state["testing_data_set"])}
        }

    return test_data_set
