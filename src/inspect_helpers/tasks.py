from inspect_ai import Task, task
from inspect_ai.solver import generate
from inspect_ai.log import read_eval_log
from inspect_ai.scorer import model_graded_fact
from src.inspect_helpers.solvers import prefill_generate
from src.inspect_helpers.datasets import create_samples_from_csv
from src.inspect_helpers.scorers import custom_match, custom_prompt_criterion_mgf
from typing import Callable


@task
def injection_consistency_and_recognition(
    csv_file_path: str,
    treatment_col: str = None,
    prompt_template_path: str = "prompts/prompt_template.txt",
    prefix_template_path: str = "prompts/prefix_template.txt",
    passage_column: str = "text",
) -> Task:
    return Task(
        dataset=create_samples_from_csv(
            csv_file_path=csv_file_path,
            treatment_col=treatment_col,
            prompt_template_path=prompt_template_path,
            prefix_template_path=prefix_template_path,
            passage_column=passage_column,
        ),
        solver=prefill_generate(),
        scorer=[
            custom_match(target="Answer - No", location="any", ignore_case=True),
            custom_prompt_criterion_mgf(criterion="None"),
        ],
    )
