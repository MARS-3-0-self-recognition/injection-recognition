"""
Rescore existing evaluation logs with consistency_scorer_mgf.
This script follows the patterns established in control_and_treatment_experiment.ipynb
"""

import asyncio
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")
    print("Continuing without .env file support...")

import pandas as pd
from inspect_ai import score_async
from inspect_ai.log import EvalLog, list_eval_logs, read_eval_log, write_eval_log
from inspect_ai.model import Model, GenerateConfig, get_model
from inspect_ai.scorer import Scorer

from src.inspect_helpers.scorers import consistency_scorer_mgf
from src.inspect_helpers.utils import collect_logs_by_model, get_validated_logs_by_model


# ============================================================================
# Configuration (following control_and_treatment_experiment.ipynb patterns)
# ============================================================================

EXPERIMENT_NAME = "wikihow_summary_injection"
CONTROL_LOG_DIR = f"logs/{EXPERIMENT_NAME}/control"
TREATMENT_LOG_DIR = f"logs/{EXPERIMENT_NAME}/treatment"
RESCORED_LOG_DIR = f"logs/{EXPERIMENT_NAME}/rescored"

# Model configurations
MODELS = [
    "anthropic/claude-sonnet-4-20250514",
    "anthropic/claude-3-5-haiku-20241022",
    # "ollama/gemma3:1b-it-q8_0",
    # "ollama/gemma3:4b-it-q8_0",
]

islocal = {
    "ollama": True,
    "together": False,
    "anthropic": False,
    "google": False,
}

BATCH_SIZE_LOCAL = 4
MAX_CONNECTIONS_API = 100

# Scoring model configuration (same as in notebook)
# For debugging: Add max_tokens limit to reduce token usage
DEBUG_MODE = True  # Set to False for production
MAX_TOKENS_DEBUG = None  # Even smaller for quick tests
LIMIT_SAMPLES_DEBUG = 1  # Process only first N samples from each log
# Note: For treatment logs, we need to process ALL logs per model, not just 1
MAX_LOGS_PER_MODEL_DEBUG = 1  # None to process all logs per model (set to int to limit)
MAX_MODELS_DEBUG = 1  # None to process all models (set to int to limit number of models)
                      # Example: Set to 1 to test on just the first model, 2 for first two models, etc.
MAX_TREATMENTS_DEBUG = 1  # None to process all treatments (set to int to limit number of treatments)
                          # Example: Set to 1 to test on just the first treatment, 2 for first two treatments, etc.
SKIP_CONTROL_DEBUG = True  # Set to True to skip control processing in debug mode

def select_default_scoring_model_name() -> str:
    """Select a scoring model based on available provider API keys."""
    if os.getenv("INSPECT_SCORING_MODEL"):
        return os.getenv("INSPECT_SCORING_MODEL")  # allow explicit override
    
    # Check for API keys and provide debug info
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    
    if DEBUG_MODE:
        print(f"DEBUG: API Key availability check:")
        print(f"  - ANTHROPIC_API_KEY: {'✓ Available' if anthropic_key else '✗ Not found'}")
        print(f"  - GOOGLE_API_KEY: {'✓ Available' if google_key else '✗ Not found'}")
    
    if anthropic_key:
        return "anthropic/claude-3-5-haiku-20241022"
    if google_key:
        return "google/gemini-2.5-flash-lite"
    # Add other providers here if desired (Together, OpenAI, etc.)
    # As a last resort, try a local model if configured (commented example):
    # return "ollama/gemma3:1b-it-q8_0"
    raise RuntimeError(
        "No scoring model available: set ANTHROPIC_API_KEY or GOOGLE_API_KEY (or INSPECT_SCORING_MODEL)."
    )


def get_scoring_model_name() -> str:
    return select_default_scoring_model_name()


# ============================================================================
# Helper Functions (following notebook patterns)
# ============================================================================

def split_provider_and_model(model: str) -> tuple[str, str]:
    """Split model string into provider and model name."""
    parts = model.split("/")
    return parts[0], parts[1]


def resolve_max_connections(model: str | Model) -> Model:
    """
    Resolve max connections for a model based on whether it's local or API.
    This follows the same pattern as in control_and_treatment_experiment.ipynb
    """
    if isinstance(model, Model):
        if model.config.max_connections is not None:
            # Apply debug token limit if needed
            if DEBUG_MODE and model.config.max_tokens is None:
                model_args = model.config.model_dump()
                model_args["max_tokens"] = MAX_TOKENS_DEBUG
                provider = split_provider_and_model(str(model))[0]
                model_args["max_connections"] = (
                    BATCH_SIZE_LOCAL if islocal.get(provider, False)
                    else MAX_CONNECTIONS_API
                )
                return get_model(
                    str(model),
                    config=GenerateConfig(**model_args),
                )
            return model
        else:
            model_args = model.config.model_dump()
            provider = split_provider_and_model(str(model))[0]
            model_args["max_connections"] = (
                BATCH_SIZE_LOCAL if islocal.get(provider, False)
                else MAX_CONNECTIONS_API
            )
            # Apply debug token limit
            if DEBUG_MODE:
                model_args["max_tokens"] = MAX_TOKENS_DEBUG
            return get_model(
                str(model),
                config=GenerateConfig(**model_args),
            )
    
    provider = split_provider_and_model(model)[0]
    config_args = {
        "max_connections": (
            BATCH_SIZE_LOCAL if islocal.get(provider, False)
            else MAX_CONNECTIONS_API
        )
    }
    # Apply debug config
    if DEBUG_MODE:
        config_args["max_tokens"] = MAX_TOKENS_DEBUG
        config_args["temperature"] = 0
        config_args["reasoning_tokens"] = -1
    else:
        # match previous non-debug default
        config_args["reasoning_tokens"] = -1
    
    return get_model(
        model,
        config=GenerateConfig(**config_args)
    )


def windows_safe_path(path: str) -> str:
    """Convert path to Windows-safe format if needed."""
    # This is a placeholder - implement based on your OS requirements
    return path


def url_to_file_path(url_or_path: str) -> str:
    """Convert file:// URL to regular file path, or return path as-is if not a URL."""
    if url_or_path.startswith('file://'):
        # Remove file:// prefix and handle Windows paths properly
        path = url_or_path[7:]  # Remove 'file://' prefix
        # On Windows, paths like '/C:/Users/...' need the leading slash removed
        if len(path) > 2 and path[0] == '/' and path[2] == ':':
            path = path[1:]  # Remove leading slash before drive letter
        return path
    return url_or_path


async def rescore_eval_log(
    log_file: str,
    scorer: Scorer,
    scorer_args: Dict[str, Any] = None,
    output_file: str = None,
    action: str = "append",
    limit_samples: int = None  # Add sample limit for debugging
) -> EvalLog:
    """
    Rescore an existing evaluation log with the consistency_scorer_mgf.
    
    Args:
        log_file: Path to the .eval file
        scorer: The scorer function to use
        scorer_args: Arguments for the scorer
        output_file: Path for output (if None, overwrites input file)
        action: Whether to "append" or "overwrite" existing scores
        limit_samples: Limit number of samples to process (for debugging)
    
    Returns:
        The rescored EvalLog
    """
    # Read the existing evaluation log
    eval_log = read_eval_log(log_file)
    
    # In debug mode, limit the samples to process
    if limit_samples and eval_log.samples:
        original_sample_count = len(eval_log.samples)
        eval_log.samples = eval_log.samples[:limit_samples]
        print(f"    DEBUG: Processing {len(eval_log.samples)} of {original_sample_count} samples")
    
    # Re-score with the new scorer
    try:
        scored_log = await score_async(
            log=eval_log,
            scorers=[scorer],
            action=action
        )
    except Exception as e:
        raise RuntimeError(f"Failed during scoring with scorer: {e}")
    
    # Write the updated log to the output file
    output_path = output_file or log_file
    try:
        # Debug: Check if directory actually exists before writing
        output_dir = os.path.dirname(output_path)
        if DEBUG_MODE:
            print(f"    DEBUG: About to write to: {output_path}")
            print(f"    DEBUG: Parent directory: {output_dir}")
            print(f"    DEBUG: Parent directory exists: {os.path.exists(output_dir)}")
            print(f"    DEBUG: Parent directory is writable: {os.access(output_dir, os.W_OK) if os.path.exists(output_dir) else 'N/A'}")
        
        # Test if we can write a simple file first
        if DEBUG_MODE:
            test_file = output_path + ".test"
            try:
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                print(f"    DEBUG: Simple file write test: SUCCESS")
            except Exception as e:
                print(f"    DEBUG: Simple file write test: FAILED - {e}")
        
        write_eval_log(scored_log, output_path)
    except Exception as e:
        raise RuntimeError(f"Failed writing to {output_path}: {e}")
    
    return scored_log


async def rescore_logs_by_model(
    log_dir: str,
    output_dir: str,
    scorer_criterion: tuple[str, str],
    experiment_name: str = None,
    max_logs_per_model: int = None,  # Add limit for debugging
    max_models: int = None  # Add limit for number of models
) -> Dict[str, List[EvalLog]]:
    """
    Rescore all evaluation logs in a directory, organized by model.
    
    Args:
        log_dir: Directory containing evaluation logs
        output_dir: Directory to save rescored logs
        scorer_criterion: Tuple of (criterion, treatment_style) for the scorer
        experiment_name: Name of the experiment (for validation)
        max_logs_per_model: Maximum number of logs to process per model (for debugging)
        max_models: Maximum number of models to process (for debugging)
    
    Returns:
        Dictionary mapping model names to lists of rescored logs
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all log files
    log_files = list_eval_logs(log_dir, filter=lambda log: log.status == "success")
    
    if experiment_name:
        # Use validation if experiment name is provided
        logs_by_model = get_validated_logs_by_model(log_dir, experiment_name)
    else:
        # Otherwise collect logs by model and filter for successful logs only
        all_logs_by_model = collect_logs_by_model(log_dir)
        logs_by_model = {}
        for model_name, logs in all_logs_by_model.items():
            successful_logs = [log for log in logs if log.get("status") == "success"]
            if successful_logs:
                logs_by_model[model_name] = successful_logs
    
    rescored_logs = {}
    
    # Apply model limit if specified
    models_to_process = list(logs_by_model.items())
    if max_models:
        models_to_process = models_to_process[:max_models]
        if DEBUG_MODE:
            print(f"DEBUG: Processing only {len(models_to_process)} of {len(logs_by_model)} models")
    
    for model_name, logs in models_to_process:
        print(f"\nProcessing model: {model_name}")
        model_rescored_logs = []
        
        # Create model-specific output directory
        model_output_dir = os.path.join(output_dir, model_name.replace("/", "_"))
        # Make absolute and normalize to forward slashes to match write_eval_log behavior
        model_output_dir = os.path.abspath(model_output_dir).replace('\\', '/')
        os.makedirs(model_output_dir, exist_ok=True)
        
        if DEBUG_MODE:
            print(f"  DEBUG: Created output directory: {model_output_dir}")
            print(f"  DEBUG: Directory exists: {os.path.exists(model_output_dir)}")
        
        # Limit logs if in debug mode
        logs_to_process = logs
        if DEBUG_MODE and max_logs_per_model:
            logs_to_process = logs[:max_logs_per_model]
            print(f"  DEBUG: Processing only {len(logs_to_process)} of {len(logs)} logs")
        
        for log_info in logs_to_process:
            # Both collect_logs_by_model and get_validated_logs_by_model return the same structure
            if isinstance(log_info, dict) and log_info.get("status") == "success":
                # Handle the dictionary format from both functions
                raw_log_path = log_info["log_info"].name
                log_path = url_to_file_path(raw_log_path)  # Convert file:// URL to file path
                eval_log = log_info["eval_log"]
            else:
                # Skip non-successful or malformed entries
                print(f"  Skipping non-successful log: {log_info}")
                continue
            
            try:
                # Generate output filename - shorten to avoid Windows path length limits
                base_name = os.path.basename(log_path)
                # Extract just the timestamp and task ID from the long filename
                import re
                match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}[^_]+).*?([A-Za-z0-9]{22})\.eval$', base_name)
                if match:
                    timestamp, task_id = match.groups()
                    short_name = f"rescored_{timestamp}_{task_id}.eval"
                else:
                    # Fallback: just use first 50 chars + task ID if we can find it
                    task_id_match = re.search(r'([A-Za-z0-9]{22})(?:_with_consistency_recognition_conf_matrix_scores)?\.eval$', base_name)
                    if task_id_match:
                        task_id = task_id_match.group(1)
                        short_name = f"rescored_{task_id}.eval"
                    else:
                        short_name = f"rescored_{base_name[:50]}.eval"
                
                output_file = os.path.join(model_output_dir, short_name)
                # Make absolute path and normalize to forward slashes to match write_eval_log behavior
                output_file = os.path.abspath(output_file).replace('\\', '/')
                
                if DEBUG_MODE:
                    print(f"    DEBUG: Converting path: {raw_log_path} -> {log_path}")
                    print(f"    DEBUG: Input file: {log_path}")
                    print(f"    DEBUG: Output file: {output_file}")
                    print(f"    DEBUG: Output dir exists: {os.path.exists(os.path.dirname(output_file))}")
                
                # Configure the scorer with the criterion
                # Lazily select and construct the scoring model only when needed
                scoring_model = resolve_max_connections(get_scoring_model_name())
                scorer = consistency_scorer_mgf(
                    criterion=scorer_criterion[1],  # Treatment style
                    model=scoring_model
                )
                
                # Rescore the log
                print(f"  Rescoring: {base_name}")
                rescored_log = await rescore_eval_log(
                    log_file=log_path,
                    scorer=scorer,
                    scorer_args={"criterion": scorer_criterion},
                    output_file=output_file,
                    action="append",
                    limit_samples=LIMIT_SAMPLES_DEBUG if DEBUG_MODE else None
                )
                
                model_rescored_logs.append(rescored_log)
                print(f"  ✓ Saved to: {output_file}")
                
            except Exception as e:
                print(f"  ✗ Error rescoring {log_path}: {e}")
        
        rescored_logs[model_name] = model_rescored_logs
    
    return rescored_logs


async def main():
    """
    Main function to rescore evaluation logs.
    Follows the same structure as control_and_treatment_experiment.ipynb
    """
    print("=" * 60)
    print("RESCORING EVALUATION LOGS")
    if DEBUG_MODE:
        print(f"DEBUG MODE ACTIVE:")
        print(f"  - Token limit: {MAX_TOKENS_DEBUG}")
        print(f"  - Sample limit per log: {LIMIT_SAMPLES_DEBUG}")
        print(f"  - Max logs per model: {MAX_LOGS_PER_MODEL_DEBUG}")
        print(f"  - Max models to process: {MAX_MODELS_DEBUG}")
        print(f"  - Max treatments to process: {MAX_TREATMENTS_DEBUG}")
        print(f"  - Skip control processing: {SKIP_CONTROL_DEBUG}")
        print(f"  - Temperature: 0 (deterministic)")
    print("=" * 60)
    
    # Define scorer criteria for different conditions
    # These should match what was used in the original evaluations
    control_criterion = ("No", "None")
    
    # Treatment criteria examples (adjust based on your needs)
    treatment_criteria = [
        ("Yes", "Capitalization"),
        ("Yes", "Typing and spelling errors"),
    ]
    
    # In debug mode, optionally limit criteria to process
    if DEBUG_MODE:
        print("\nDEBUG: Processing limited subset for testing")
        if MAX_TREATMENTS_DEBUG:
            original_treatment_count = len(treatment_criteria)
            treatment_criteria = treatment_criteria[:MAX_TREATMENTS_DEBUG]
            print(f"DEBUG: Processing only {len(treatment_criteria)} of {original_treatment_count} treatments")
    
    # Process control logs (unless skipped in debug mode)
    control_rescored = {}
    if not (DEBUG_MODE and SKIP_CONTROL_DEBUG):
        print("\n--- Processing Control Logs ---")
        control_rescored = await rescore_logs_by_model(
            log_dir=CONTROL_LOG_DIR,
            output_dir=os.path.join(RESCORED_LOG_DIR, "control"),
            scorer_criterion=control_criterion,
            experiment_name=EXPERIMENT_NAME,
            max_logs_per_model=MAX_LOGS_PER_MODEL_DEBUG if DEBUG_MODE else None,
            max_models=MAX_MODELS_DEBUG if DEBUG_MODE else None
        )
    else:
        print("\n--- Skipping Control Logs (DEBUG_MODE with SKIP_CONTROL_DEBUG=True) ---")
    
    # Process treatment logs
    print("\n--- Processing Treatment Logs ---")
    for criterion in treatment_criteria:
        treatment_name = criterion[1].lower().replace(" ", "_")
        print(f"\nProcessing treatment: {treatment_name}")
        
        treatment_rescored = await rescore_logs_by_model(
            log_dir=TREATMENT_LOG_DIR,
            output_dir=os.path.join(RESCORED_LOG_DIR, f"treatment_{treatment_name}"),
            scorer_criterion=criterion,
            # Multiple successful logs per model are expected in treatment; skip uniqueness validation
            experiment_name=None,
            max_logs_per_model=MAX_LOGS_PER_MODEL_DEBUG if DEBUG_MODE else None,
            max_models=MAX_MODELS_DEBUG if DEBUG_MODE else None
        )
    
    # Generate summary DataFrame (similar to notebook's evals_df usage)
    print("\n--- Generating Summary ---")
    
    # Collect all rescored logs
    all_rescored_logs = []
    rescored_dirs = []
    
    # Add control directory only if it was processed
    if not (DEBUG_MODE and SKIP_CONTROL_DEBUG):
        rescored_dirs.append(os.path.join(RESCORED_LOG_DIR, "control"))
    
    # Add treatment directories
    rescored_dirs.extend([
        os.path.join(RESCORED_LOG_DIR, f"treatment_{c[1].lower().replace(' ', '_')}") 
        for c in treatment_criteria
    ])
    
    for rescored_dir in rescored_dirs:
        if os.path.exists(rescored_dir):
            rescored_files = list_eval_logs(rescored_dir)
            all_rescored_logs.extend(rescored_files)
    
    if all_rescored_logs:
        # Import only if we have logs to process
        from inspect_ai.analysis import evals_df
        
        # Create DataFrame of rescored evaluations
        rescored_df = evals_df(all_rescored_logs)
        
        # Save summary to CSV
        summary_path = os.path.join(RESCORED_LOG_DIR, "rescored_summary.csv")
        rescored_df.to_csv(summary_path, index=False)
        print(f"\n✓ Summary saved to: {summary_path}")
        
        # Display basic statistics
        print("\nRescoring Summary:")
        print(f"  Total rescored logs: {len(rescored_df)}")
        print(f"  Models processed: {rescored_df['model'].nunique() if 'model' in rescored_df else 'N/A'}")
        
        # Display score columns if available
        score_cols = [col for col in rescored_df.columns if col.startswith('score_')]
        if score_cols:
            print("\n  Available scores:")
            for col in score_cols:
                print(f"    - {col}")
    
    print("\n" + "=" * 60)
    print("RESCORING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())