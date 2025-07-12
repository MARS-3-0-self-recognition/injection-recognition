import csv
import os
from typing import List
from inspect_ai.dataset import Sample, MemoryDataset
import urllib.parse

PREFILL_KEY = "prefill"
ROW_INDEX_KEY = "row_index"

def load_prompt_template(template_path: str) -> str:
    """Load a prompt template from a file."""
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_prefill_template(template_path: str) -> str:

    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()


def parse_treatment_columns(csv_file_path: str) -> List[str]:
    """
    Parse CSV file to identify treatment columns.
    Treatment columns are those that start with 'IL' followed by percentage, comma, strength, and number.
    """
    treatment_columns = []
    
    with open(csv_file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        
        for header in headers:
            if header and header.startswith('IL'):
                treatment_columns.append(header)
    
    return treatment_columns


def safe_url_encode(text: str) -> str:
    """
    Safely encode text for use in URLs/IDs to prevent URI malformed errors.
    Replaces problematic characters with URL-safe alternatives.
    """
    return urllib.parse.quote(text, safe='')


def create_samples_from_csv(
    csv_file_path: str,
    treatment_col: str = None,
    default_prefill: str = "",
    passage_column: str = "text",
    prompt_template_path: str = "prompts/prompt_template.txt",
    prefill_template_path: str = "prompts/prefix_template.txt",
) -> List[Sample]:
    """
    Create Inspect AI samples from CSV file with treatments as pre-fills.
    
    Args:
        csv_file_path: Path to the CSV file containing treatments and passages
        treatment_col: Name of the column containing treatment values
        prompt_template_path: Path to the prompt template file
        prefill_template_path: Path to the prefix template file
        passage_column: Name of the column containing passage text
    
    Returns:
        List of Sample objects for Inspect AI evaluation
    """
    samples = []
    
    # Load templates
    prompt_template = load_prompt_template(prompt_template_path)
    prefill_template = load_prefill_template(prefill_template_path)
    
    # Read CSV data
    with open(csv_file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row_idx, row in enumerate(reader):
            passage = row.get(passage_column, "")

            # Format the prompt with the passage
            formatted_prompt = prompt_template.format(passage=passage)
            
            def get_prefill(treatment_col: str | None) -> tuple[str, str]:
                if treatment_col is None:
                    return "", default_prefill
                prefill = row.get(treatment_col, "")
                
                formatted_prefill = prefill_template.format(prefill=prefill)
                # URL-encode the treatment column name for safe use in IDs
                safe_treatment_col = safe_url_encode(treatment_col)
                
                return safe_treatment_col, formatted_prefill

            safe_treatment_col, formatted_prefill = get_prefill(treatment_col)
            # Create the sample
            sample = Sample(
                input=formatted_prompt,
                metadata={
                    ROW_INDEX_KEY: row_idx,
                    PREFILL_KEY: formatted_prefill,
                    "treatment_column": safe_treatment_col,
                    "passage": passage,
                    "csv_file": os.path.basename(csv_file_path)
                }
            )
            
            samples.append(sample)
    
    return MemoryDataset(samples)