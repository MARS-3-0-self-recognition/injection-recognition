import csv
import os
from typing import List
from inspect_ai.dataset import Sample, MemoryDataset
import urllib.parse


def load_prompt_template(template_path: str) -> str:
    """Load a prompt template from a file."""
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_prefix_template(template_path: str) -> str:
    """Load a prefix template from a file."""
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
    treatment_col: str,
    prompt_template_path: str = "prompts/prompt_template.txt",
    prefix_template_path: str = "prompts/prefix_template.txt",
    passage_column: str = "text"
) -> List[Sample]:
    """
    Create Inspect AI samples from CSV file with treatments as pre-fill injections.
    
    Args:
        csv_file_path: Path to the CSV file containing treatments and passages
        treatment_col: Name of the column containing treatment values
        prompt_template_path: Path to the prompt template file
        prefix_template_path: Path to the prefix template file
        passage_column: Name of the column containing passage text
    
    Returns:
        List of Sample objects for Inspect AI evaluation
    """
    samples = []
    
    # Load templates
    prompt_template = load_prompt_template(prompt_template_path)
    prefix_template = load_prefix_template(prefix_template_path)
    
    # Read CSV data
    with open(csv_file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row_idx, row in enumerate(reader):
            passage = row.get(passage_column, "")
            
            injection = row.get(treatment_col, "")
            
            if injection:  # Only create sample if treatment has a value
                # Format the prompt with the passage
                formatted_prompt = prompt_template.format(passage=passage)
                
                # Format the prefix with the injection
                formatted_prefix = prefix_template.format(injection=injection)
                
                # URL-encode the treatment column name for safe use in IDs
                safe_treatment_col = safe_url_encode(treatment_col)
                
                # Create the sample
                sample = Sample(
                    input=formatted_prompt,
                    target="YES",
                    id=f"row_{row_idx}_{safe_treatment_col}",
                    metadata={
                        "row_index": row_idx,
                        "treatment_column": treatment_col,  # Keep original for reference
                        "treatment_column_encoded": safe_treatment_col,  # Add encoded version
                        "injection": injection,
                        "passage": passage,
                        "csv_file": os.path.basename(csv_file_path)
                    }
                )
                
                # Add the prefix as a system message or pre-fill
                # This will be handled by the solver configuration
                sample.metadata["prefix"] = formatted_prefix
                
                samples.append(sample)
    
    return MemoryDataset(samples)