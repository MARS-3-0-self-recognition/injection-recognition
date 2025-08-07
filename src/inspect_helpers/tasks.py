from inspect_ai import Task, task
from inspect_ai.solver import generate
from inspect_ai.log import read_eval_log
from inspect_ai.scorer import model_graded_fact
from inspect_ai.model import Model
from src.inspect_helpers.solvers import prefill_generate
from src.inspect_helpers.datasets import create_samples_from_csv
from src.inspect_helpers.scorers import answer_match, which_treatment_mgf
from typing import Callable
# TODO: remove the hardcoded custom match target template i.e. "Answer - "


@task
def injection_consistency_and_recognition(
    csv_file_path: str,
    treatment_col: str | None,
    scorer_criteria: tuple[str, str],
    default_prefill: str = "Task 1:",
    prompt_template_path: str = "prompts/prompt_template.txt",
    prompt_template_args: dict[str, str] = {},
    prefill_template_path: str = "prompts/prefill_template.txt",
    passage_column: str = "text",
    task_model: str | Model | None = None,
    scorer_model: str | Model | None = None,
) -> Task:
    # if isinstance(scorer_criteria, str):
    #     if scorer_criteria not in ["Yes", "No"]:
    #         match_criteria = "No" if scorer_criteria == "None" else "Yes"
    #         Warning(
    #             f"Match scorer criteria expects 'Yes' or 'No', got {scorer_criteria}. Converting to {match_criteria}"
    #         )
    #         scorer_criteria = match_criteria
    
    scorers = [
        answer_match(
            target=f"Answer - {scorer_criteria[0]}",
            location="any",
            ignore_case=True,
        ),
        which_treatment_mgf(
            criterion=scorer_criteria[1],
            model=scorer_model,
        ),
        # text_until_last_error(),
    ]
    # if "typo" in csv_file_path:
    #     scorers.append(text_until_last_typo())
        
    return Task(
        dataset=create_samples_from_csv(
            csv_file_path=csv_file_path,
            treatment_col=treatment_col,
            default_prefill=default_prefill,
            prompt_template_args=prompt_template_args,
            prompt_template_path=prompt_template_path,
            prefill_template_path=prefill_template_path,
            passage_column=passage_column,
        ),
        solver=prefill_generate(default_prefill),
        scorer=scorers,
        model=task_model,
    )
