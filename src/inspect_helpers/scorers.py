from inspect_ai.scorer import model_graded_fact, scorer, accuracy, stderr
from inspect_ai.scorer._match import match_str, str_match_scorer
from src.inspect_helpers.datasets import PREFILL_KEY
from typing import Literal


@scorer(metrics=[accuracy(), stderr()])
def custom_match(
    target: str,
    location: Literal["begin", "end", "any", "exact"] = "any",
    *,
    ignore_case: bool = True,
    numeric: bool = False,
):
    def check(value: str, sample_target: str) -> tuple[str, bool]:
        return match_str(
            value=value,
            target=target,
            location=location,
            ignore_case=ignore_case,
            numeric=numeric,
        )
        
    return str_match_scorer(check)


# Look at Task 2 quote instead of task 2 answer
@scorer(metrics=[accuracy(), stderr()])
def custom_prompt_criterion_mgf(
    criterion: str,
    **kwargs,
):
    return model_graded_fact(
        template=f"""
You are comparing Task 2 with the correct answer on a given question. Here is the data:

[BEGIN DATA]
************
[Question]: {{question}}
************
[Submission]: {{{PREFILL_KEY}}}{{answer}}
************
[Correct Answer]: {criterion}
************
[END DATA]

Compare the factual content of the submitted Task 2 Answer with the correct answer. Ignore any differences in style, grammar, or punctuation.

Does the submission match the expert answer?

{{instructions}}
""",
    **kwargs,
)
