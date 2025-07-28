from inspect_ai.log import list_eval_logs, read_eval_log

def collect_logs_by_model(log_dir):
    """
    Collect all evaluation logs from a directory and group them by model.

    Args:
        log_dir: Directory containing evaluation logs

    Returns:
        Dictionary mapping model names to lists of log data
    """
    logs_by_model = {}

    for eval_log_info in list_eval_logs(log_dir):
        eval_log = read_eval_log(eval_log_info)
        model_name = eval_log.eval.model.replace("/", "_")

        if model_name not in logs_by_model:
            logs_by_model[model_name] = []

        logs_by_model[model_name].append(
            {"log_info": eval_log_info, "eval_log": eval_log, "status": eval_log.status}
        )

    return logs_by_model


def validate_single_successful_log_per_model(logs_by_model, experiment_name=None):
    """
    Validate that there's only one successful log per model.

    Args:
        logs_by_model: Dictionary mapping model names to lists of log data
        experiment_name: Optional experiment name for better error messages

    Raises:
        ValueError: If multiple successful logs found for the same model
    """
    errors = []
    
    for model_name, logs in logs_by_model.items():
        successful_logs = [log for log in logs if log["status"] == "success"]

        if len(successful_logs) > 1:
            experiment_prefix = (
                f" in experiment '{experiment_name}'" if experiment_name else ""
            )
            log_files = [log["log_info"].name for log in successful_logs]
            errors.append(
                f"Model '{model_name}' has {len(successful_logs)} successful logs for {experiment_prefix}:\n"
                f"  Files: {', '.join(log_files)}\n"
                f"  Please remove duplicate logs or use only one successful run per model."
            )
        elif len(successful_logs) == 0:
            experiment_prefix = (
                f" in experiment '{experiment_name}'" if experiment_name else ""
            )
            print(
                f"Warning: Model '{model_name}' has no successful logs{experiment_prefix}"
            )

    if errors:
        raise ValueError(
            "Multiple successful logs found for some models:\n\n" + "\n\n".join(errors)
        )


def get_validated_logs_by_model(log_dir, experiment_name=None):
    """
    Get logs grouped by model after validating there's only one successful log per model.

    Args:
        log_dir: Directory containing evaluation logs
        experiment_name: Optional experiment name for better error messages

    Returns:
        Dictionary mapping model names to lists of log data

    Raises:
        ValueError: If multiple successful logs found for the same model
    """
    logs_by_model = collect_logs_by_model(log_dir)
    validate_single_successful_log_per_model(logs_by_model, experiment_name)
    return logs_by_model