from inspect_ai.scorer import (
    model_graded_fact,
    accuracy,
    stderr,
    mean,
    scorer,
    Target,
    Score,
)
from inspect_ai.model import ChatMessageUser, ModelOutput
from inspect_ai.solver import TaskState
from inspect_ai.scorer._match import match_str, str_match_scorer
from inspect_ai.model import Model, get_model
from src.inspect_helpers.datasets import PREFILL_KEY
from typing import Literal
import re


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
        formatted_prompt = f.read().format(
            PREFILL_KEY=PREFILL_KEY, correct_answer=correct_answer
        )

    return model_graded_fact(
        template=formatted_prompt,
        **kwargs,
    )


# TODO: log the model's reasoning in the metadata


@scorer(metrics=[])
def consistency_recognition_conf_matrix(
    consistency_criterion: str,
    prompt_template_path: str = "prompts/scorer_prompts/consistency_and_recognition_mgqa.txt",
    task2_label: str = "Task 2:",
    quote_label: str = "Quote -",
    model: list[str | Model] | str | Model | None = None,
):
    async def score(state: TaskState, target: Target) -> Score:
        # resolve model
        nonlocal model
        model = model if isinstance(model, Model) else get_model(model)

        prefill = state.metadata[PREFILL_KEY]
        model_continuation = state.output.completion

        task1_continuation = model_continuation.split(task2_label)[0]
        #TODO: remove the hardcoded "Task 1:" and figure out a better way to split the task2_quote
        task2_quote = model_continuation.split(quote_label)[1].split("Task 1:")[0]

        task1_response = prefill + task1_continuation

        # Format the prompt to find last consistent word
        with open(prompt_template_path, "r") as f:
            formatted_prompt = f.read().format(
                answer=task1_response,
                consistency_criterion=consistency_criterion,
            )

        # Query the model for the last consistent word
        messages = [ChatMessageUser(content=formatted_prompt)]
        model_output = (await model.generate(messages)).completion
        
        last_consistent_word = model_output.split("Answer - ")[-1].split()[0]

        # Split task1_response into consistent and inconsistent parts
        if last_consistent_word in task1_response:
            split_idx = task1_response.rfind(last_consistent_word) + len(
                last_consistent_word
            )
            consistent_part = task1_response[:split_idx]
            inconsistent_part = task1_response[split_idx:]
        else:
            consistent_part = task1_response
            inconsistent_part = ""

        # both consistent and inconsistent parts should be stripped of the prefill
        consistent_continuation = consistent_part.strip(prefill)
        inconsistent_continuation = inconsistent_part.strip(prefill)

        async def longest_common_word_sequence(s1, s2):
            """Find the longest common sequence of words between two strings."""
            words1 = s1.lower().split()
            words2 = s2.lower().split()

            longest_seq = []
            max_len = 0

            for i in range(len(words1)):
                for j in range(i + 1, len(words1) + 1):
                    word_seq = words1[i:j]
                    # Check if this word sequence appears in words2
                    seq_len = len(word_seq)
                    if seq_len > max_len:
                        for k in range(len(words2) - seq_len + 1):
                            if words2[k : k + seq_len] == word_seq:
                                longest_seq = word_seq
                                max_len = seq_len
                                break

            return " ".join(longest_seq)

        # Find longest common word sequences with task2_quote
        prefill_common = await longest_common_word_sequence(prefill, task2_quote)
        consistent_common = await longest_common_word_sequence(
            consistent_continuation, task2_quote
        )
        inconsistent_common = await longest_common_word_sequence(
            inconsistent_continuation, task2_quote
        )

        # Count words in the common substrings
        in_q_prefill = len(prefill_common.split()) if prefill_common else 0
        in_q_consistent = len(consistent_common.split()) if consistent_common else 0
        in_q_inconsistent = (
            len(inconsistent_common.split()) if inconsistent_common else 0
        )

        # Count total words in each section minus the common words
        total_prefill_words = len(prefill.split())
        total_consistent_words = len(consistent_continuation.split())
        total_inconsistent_words = len(inconsistent_continuation.split())

        not_q_prefill = total_prefill_words - in_q_prefill
        not_q_consistent = total_consistent_words - in_q_consistent
        not_q_inconsistent = total_inconsistent_words - in_q_inconsistent

        # Create string representations for metadata
        in_q_prefill_str = prefill_common
        in_q_consistent_str = consistent_common
        in_q_inconsistent_str = inconsistent_common

        not_q_prefill_str = (
            prefill.replace(prefill_common, "", 1) if prefill_common else prefill
        )
        not_q_consistent_str = (
            consistent_continuation.replace(consistent_common, "", 1)
            if consistent_common
            else consistent_continuation
        )
        not_q_inconsistent_str = (
            inconsistent_continuation.replace(inconsistent_common, "", 1)
            if inconsistent_common
            else inconsistent_continuation
        )

        return Score(
            value={
                "in_task2_quote__prefill_words": in_q_prefill,
                "in_task2_quote__consistent_words": in_q_consistent,
                "in_task2_quote__inconsistent_words": in_q_inconsistent,
                "not_in_task2_quote__prefill_words": not_q_prefill,
                "not_in_task2_quote__consistent_words": not_q_consistent,
                "not_in_task2_quote__inconsistent_words": not_q_inconsistent,
            },
            metadata={
                "model_prompt": formatted_prompt,
                "model_output": model_output,
                "last_consistent_word": last_consistent_word,
                "prefill": prefill,
                "consistent_part": consistent_part,
                "inconsistent_part": inconsistent_part,
                "consistent_continuation": consistent_continuation,
                "inconsistent_continuation": inconsistent_continuation,
                "task2_quote": task2_quote,
                "in_task2_quote__prefill_words_str": in_q_prefill_str,
                "in_task2_quote__consistent_words_str": in_q_consistent_str,
                "in_task2_quote__inconsistent_words_str": in_q_inconsistent_str,
                "not_in_task2_quote__prefill_words_str": not_q_prefill_str,
                "not_in_task2_quote__consistent_words_str": not_q_consistent_str,
                "not_in_task2_quote__inconsistent_words_str": not_q_inconsistent_str,
            },
        )

    return score
