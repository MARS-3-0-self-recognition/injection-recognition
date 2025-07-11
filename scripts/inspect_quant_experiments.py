from src.inspect_helpers.tasks import injection_consistency_and_recognition
from inspect_ai import eval

eval(
    tasks=[
        injection_consistency_and_recognition(
            csv_file_path="data/wikisum_capitalization_treatments_6_10.csv",
            treatment_col=treatment_col
        )
        for treatment_col in ["IL33_S1", "IL33_S2"]
    ],
    model=["ollama/qwen3:0.6b", "anthropic/claude-sonnet-4-20250514"],
    limit=2,
    log_dir="logs/wikisum_capitalization_treatments",
    max_connections=100, 
    timeout=500,
)