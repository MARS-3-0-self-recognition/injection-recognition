from inspect_ai.scorer import model_graded_fact, scorer, accuracy, stderr
from inspect_ai.scorer._match import match_str, str_match_scorer
from src.inspect_helpers.datasets import PREFILL_KEY
from typing import Literal
import os


@scorer(metrics=[accuracy(), stderr()])
def answer_match(
    correct_answer: str,
    match_template_path: str = "prompts/scorer_prompts/answer_match.txt",
    location: Literal["begin", "end", "any", "exact"] = "any",
    *,
    ignore_case: bool = True,
    numeric: bool = False,
):
    with open(match_template_path, "r") as f:
        match_string = f.read().format(correct_answer=correct_answer)

    def check(value: str, sample_target: str) -> tuple[str, bool]:
        return match_str(
            value=value,
            target=match_string,
            location=location,
            ignore_case=ignore_case,
            numeric=numeric,
        )

    return str_match_scorer(check)


@scorer(metrics=[accuracy(), stderr()])
def which_treatment_mgf(
    correct_answer: str,
    prompt_template_path: str = "prompts/scorer_prompts/which_treatment_mgf.txt",
    **kwargs,
):
    # Read prompt template from file
    with open(prompt_template_path, "r") as f:
        template = f.read().format(PREFILL_KEY=PREFILL_KEY, correct_answer=correct_answer)

    return model_graded_fact(
        template=template,
        **kwargs,
    )
