#!/usr/bin/env python3
"""
Simplified WikiSum dataset utilities.

This module provides simplified functions for:
1. Getting WikiSum data by index range
2. Applying treatments to the data
"""

import pandas as pd
from pathlib import Path
from typing import List, Optional, Union
import csv

# Add the current directory to the Python path to import our modules
import sys
sys.path.append(str(Path(__file__).parent))

from string_modifier import randomly_capitalize_string
from wiki_article_processor import WikiArticleProcessor


def get_WikiSum(start_idx: int, end_idx: int, use_huggingface: bool = True) -> pd.DataFrame:
    """
    Get WikiSum articles by index range and return as DataFrame.
    
    Args:
        start_idx (int): Starting index (inclusive)
        end_idx (int): Ending index (exclusive)
        use_huggingface (bool): If True, load from Hugging Face datasets library
        
    Returns:
        pd.DataFrame: DataFrame containing the requested articles
        
    Raises:
        ValueError: If invalid range or no articles found
    """
    # Validate range
    if start_idx < 0 or end_idx <= start_idx:
        raise ValueError(f"Invalid range: start_idx={start_idx}, end_idx={end_idx}")
    
    # Create filename based on range
    filename = f"wikisum_{start_idx}_{end_idx}.csv"
    
    # Ensure output directory exists
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    file_path = output_dir / filename
    
    # Check if file already exists
    if file_path.exists():
        print(f"Loading existing CSV: {file_path}")
        return pd.read_csv(file_path)
    
    # Load from WikiSum dataset
    print(f"Loading WikiSum articles {start_idx} to {end_idx-1}...")
    processor = WikiArticleProcessor()
    
    if use_huggingface:
        articles = processor.load_wikisum_dataset(use_huggingface=True)
    else:
        articles = processor.load_wikisum_dataset()
    
    if not articles:
        raise ValueError("No articles found in WikiSum dataset")
    
    # Validate range against available articles
    if end_idx > len(articles):
        print(f"Warning: end_idx {end_idx} exceeds available articles ({len(articles)}). Using {len(articles)} instead.")
        end_idx = len(articles)
    
    # Extract requested range
    selected_articles = articles[start_idx:end_idx]
    
    if not selected_articles:
        raise ValueError(f"No articles found in range {start_idx} to {end_idx}")
    
    # Convert to DataFrame
    df = pd.DataFrame(selected_articles)
    
    # Save to CSV
    print(f"Saving to CSV: {file_path}")
    df.to_csv(file_path, index=False)
    
    print(f"Loaded {len(df)} articles")
    return df


def truncate_text(text: str, percentage: int) -> str:
    """
    Truncate text to a specified percentage of its original length.
    
    Args:
        text (str): The text to truncate
        percentage (int): Percentage of original length to keep (1-100)
    
    Returns:
        str: Truncated text
    """
    if not text:
        return ""
    
    target_length = int(len(text) * percentage / 100)
    if target_length <= 0:
        return ""
    
    # Try to truncate at word boundaries
    if target_length < len(text):
        truncated = text[:target_length]
        # Find the last space to avoid cutting words in half
        last_space = truncated.rfind(' ')
        if last_space > target_length * 0.8:  # Only use space if it's not too far back
            truncated = truncated[:last_space]
        return truncated
    else:
        return text


def apply_treatments(df: pd.DataFrame, 
                    summary_lengths: List[int] = [33, 66, 100],
                    treatment_params: Optional[dict] = None) -> pd.DataFrame:
    """
    Apply treatments to the summary column of a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame with 'summary' column
        summary_lengths (List[int]): List of summary length percentages (e.g., [33, 66, 100])
        treatment_params (Optional[dict]): Dictionary of treatment parameters
            - 'capitalization_rates': List[int] - Capitalization percentages (e.g., [25, 50, 75, 100])
            - 'typo_rates': Union[List[int], dict] - Typo introduction rates
                * List[int]: Simple format - applies to substitute_rate only (e.g., [5, 10, 15])
                * dict: Granular format with specific rates for each typo type:
                    - 'substitute_rate': List[int] - Character substitution rates
                    - 'flip_rate': List[int] - Adjacent letter flip rates  
                    - 'drop_rate': List[int] - Character drop rates
                    - 'add_rate': List[int] - Character addition rates
            - 'other_treatments': dict - Any other treatment parameters
    
    Returns:
        pd.DataFrame: DataFrame with treatment columns added
    """
    # Check if summary column exists
    if 'summary' not in df.columns:
        raise ValueError("DataFrame must contain 'summary' column")
    
    # Default treatment parameters
    if treatment_params is None:
        treatment_params = {
            'capitalization_rates': [25, 50, 75, 100]
        }
    
    # Create a copy to avoid modifying original
    result_df = df.copy()
    
    # Generate IL columns (summary lengths)
    print(f"Generating summary length columns: {summary_lengths}")
    for length_pct in summary_lengths:
        il_key = f"IL{length_pct}"
        result_df[il_key] = df['summary'].apply(lambda x: truncate_text(str(x), length_pct))
    
    # Apply capitalization treatments
    if 'capitalization_rates' in treatment_params:
        cap_rates = treatment_params['capitalization_rates']
        print(f"Generating capitalization treatments: {cap_rates}")
        
        for length_pct in summary_lengths:
            il_text = result_df[f"IL{length_pct}"]
            for cap_rate in cap_rates:
                treatment_key = f"IL{length_pct}_S{cap_rate//25}"  # S1=25%, S2=50%, etc.
                result_df[treatment_key] = il_text.apply(
                    lambda x: randomly_capitalize_string(str(x), cap_rate)
                )
    
    # Apply typo treatments if specified
    if 'typo_rates' in treatment_params:
        from string_modifier import introduce_typos
        typo_rates = treatment_params['typo_rates']
        print(f"Generating typo treatments: {typo_rates}")
        
        for length_pct in summary_lengths:
            il_text = result_df[f"IL{length_pct}"]
            
            # Handle different typo rate formats
            if isinstance(typo_rates, list):
                # Simple list format - apply each rate to substitute_rate only
                for i, typo_rate in enumerate(typo_rates):
                    treatment_key = f"IL{length_pct}_S{i+1}"  # S1, S2, S3...
                    result_df[treatment_key] = il_text.apply(
                        lambda x: introduce_typos(str(x), substitute_rate=typo_rate)
                    )
            elif isinstance(typo_rates, dict):
                # Detailed dict format - apply specific rates for each typo type
                for treatment_name, rates in typo_rates.items():
                    if isinstance(rates, dict):
                        # Individual parameters for each typo type
                        for rate_name, rate_value in rates.items():
                            treatment_key = f"IL{length_pct}_{treatment_name}_{rate_name}_{rate_value}"
                            result_df[treatment_key] = il_text.apply(
                                lambda x: introduce_typos(str(x), **{rate_name: rate_value})
                            )
                    elif isinstance(rates, list):
                        # List of rates for a specific typo type
                        for i, rate_value in enumerate(rates):
                            treatment_key = f"IL{length_pct}_S{i+1}"  # S1, S2, S3...
                            result_df[treatment_key] = il_text.apply(
                                lambda x: introduce_typos(str(x), **{treatment_name: rate_value})
                            )
    
    # Apply any other custom treatments
    if 'other_treatments' in treatment_params:
        other_treatments = treatment_params['other_treatments']
        print(f"Applying custom treatments: {list(other_treatments.keys())}")
        
        # This is a placeholder for any additional treatments you might want to add
        for treatment_name, treatment_func in other_treatments.items():
            if callable(treatment_func):
                for length_pct in summary_lengths:
                    il_text = result_df[f"IL{length_pct}"]
                    treatment_key = f"IL{length_pct}_{treatment_name}"
                    result_df[treatment_key] = il_text.apply(treatment_func)
    
    # Print summary
    original_cols = len(df.columns)
    new_cols = len(result_df.columns) - original_cols
    print(f"Added {new_cols} treatment columns")
    print(f"Total columns: {len(result_df.columns)}")
    
    return result_df


def save_treated_data(df: pd.DataFrame, 
                     start_idx: int, 
                     end_idx: int, 
                     treatment_name: str = "treatments") -> str:
    """
    Save treated DataFrame to CSV with descriptive filename.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        start_idx (int): Starting index used for original data
        end_idx (int): Ending index used for original data
        treatment_name (str): Name to include in filename
        
    Returns:
        str: Path to saved file
    """
    # Ensure output directory exists
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Create filename
    filename = f"wikisum_{start_idx}_{end_idx}_{treatment_name}.csv"
    file_path = output_dir / filename
    
    # Save to CSV
    df.to_csv(file_path, index=False)
    print(f"Saved treated data to: {file_path}")
    
    return str(file_path)


def apply_treatments_separate(df: pd.DataFrame, 
                            summary_lengths: List[int] = [33, 66, 100],
                            treatment_params: Optional[dict] = None,
                            start_idx: int = 0,
                            end_idx: int = 0) -> dict:
    """
    Apply treatments to the summary column and create separate CSV files for each treatment type.
    
    Args:
        df (pd.DataFrame): Input DataFrame with 'summary' column
        summary_lengths (List[int]): List of summary length percentages (e.g., [33, 66, 100])
        treatment_params (Optional[dict]): Dictionary of treatment parameters
            - 'capitalization_rates': List[int] - Capitalization percentages (e.g., [25, 50, 75, 100])
            - 'typo_rates': Union[List[int], dict] - Typo introduction rates
                * List[int]: Simple format - applies to substitute_rate only (e.g., [5, 10, 15])
                * dict: Granular format with specific rates for each typo type:
                    - 'substitute_rate': List[int] - Character substitution rates
                    - 'flip_rate': List[int] - Adjacent letter flip rates  
                    - 'drop_rate': List[int] - Character drop rates
                    - 'add_rate': List[int] - Character addition rates
                * dict: Combined format with predefined typo combinations:
                    - 'light': dict - Light typo combination (e.g., {'substitute_rate': 5, 'flip_rate': 3})
                    - 'medium': dict - Medium typo combination
                    - 'heavy': dict - Heavy typo combination
                * dict: Per-word typo format:
                    - 'typos_per_word': float - Number of typos per word (e.g., 0.5, 1.0, 2.0)
                    - 'typo_types': set - Set of typo types to use (e.g., {'drop_rate', 'add_rate'})
            - 'other_treatments': dict - Any other treatment parameters
        start_idx (int): Starting index for filename
        end_idx (int): Ending index for filename
    
    Returns:
        dict: Dictionary mapping treatment names to their output file paths
    """
    # Check if summary column exists
    if 'summary' not in df.columns:
        raise ValueError("DataFrame must contain 'summary' column")
    
    # Default treatment parameters
    if treatment_params is None:
        treatment_params = {
            'capitalization_rates': [25, 50, 75, 100]
        }
    
    # Ensure output directory exists
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Generate IL columns (summary lengths) - these will be in all files
    print(f"Generating summary length columns: {summary_lengths}")
    base_df = df.copy()
    for length_pct in summary_lengths:
        il_key = f"IL{length_pct}"
        base_df[il_key] = df['summary'].apply(lambda x: truncate_text(str(x), length_pct))
    
    output_files = {}
    
    # Process each treatment type separately
    for treatment_name, treatment_values in treatment_params.items():
        print(f"\nProcessing {treatment_name}...")
        
        # Create a copy of base DataFrame for this treatment
        treatment_df = base_df.copy()
        
        if treatment_name == 'capitalization_rates':
            # Apply capitalization treatments
            cap_rates = treatment_values
            print(f"Generating capitalization treatments: {cap_rates}")
            
            for length_pct in summary_lengths:
                il_text = treatment_df[f"IL{length_pct}"]
                for cap_rate in cap_rates:
                    treatment_key = f"IL{length_pct}_S{cap_rate//25}"  # S1=25%, S2=50%, etc.
                    treatment_df[treatment_key] = il_text.apply(
                        lambda x: randomly_capitalize_string(str(x), cap_rate)
                    )
        
        elif treatment_name == 'typo_rates':
            # Apply typo treatments
            from string_modifier import introduce_typos, introduce_typos_per_word
            typo_rates = treatment_values
            print(f"Generating typo treatments: {typo_rates}")
            
            for length_pct in summary_lengths:
                il_text = treatment_df[f"IL{length_pct}"]
                
                # Handle different typo rate formats
                if isinstance(typo_rates, list):
                    # Simple list format - apply each rate to substitute_rate only
                    for i, typo_rate in enumerate(typo_rates):
                        treatment_key = f"IL{length_pct}_S{i+1}"  # S1, S2, S3...
                        treatment_df[treatment_key] = il_text.apply(
                            lambda x: introduce_typos(str(x), substitute_rate=typo_rate)
                        )
                elif isinstance(typo_rates, dict):
                    # Handle different dict formats
                    for typo_key, rates in typo_rates.items():
                        if isinstance(rates, list):
                            # List of rates for a specific typo type
                            for i, rate_value in enumerate(rates):
                                treatment_key = f"IL{length_pct}_S{i+1}"  # S1, S2, S3...
                                treatment_df[treatment_key] = il_text.apply(
                                    lambda x: introduce_typos(str(x), **{typo_key: rate_value})
                                )
                        elif isinstance(rates, dict):
                            # Check if this is the new per-word format
                            if 'typos_per_word' in rates:
                                # Per-word typo format
                                typos_per_word = rates['typos_per_word']
                                typo_types = rates.get('typo_types', None)
                                treatment_key = f"IL{length_pct}_{typo_key}"
                                treatment_df[treatment_key] = il_text.apply(
                                    lambda x: introduce_typos_per_word(str(x), typos_per_word, typo_types)
                                )
                            else:
                                # Combined typo parameters (e.g., 'light', 'medium', 'heavy')
                                treatment_key = f"IL{length_pct}_{typo_key}"
                                treatment_df[treatment_key] = il_text.apply(
                                    lambda x: introduce_typos(str(x), **rates)
                                )
        
        elif treatment_name == 'other_treatments':
            # Apply custom treatments
            other_treatments = treatment_values
            print(f"Applying custom treatments: {list(other_treatments.keys())}")
            
            for custom_name, treatment_func in other_treatments.items():
                if callable(treatment_func):
                    for length_pct in summary_lengths:
                        il_text = treatment_df[f"IL{length_pct}"]
                        treatment_key = f"IL{length_pct}_{custom_name}"
                        treatment_df[treatment_key] = il_text.apply(treatment_func)
        
        # Save this treatment to a separate CSV
        filename = f"wikisum_{start_idx}_{end_idx}_{treatment_name}.csv"
        file_path = output_dir / filename
        treatment_df.to_csv(file_path, index=False)
        
        # Store the file path
        output_files[treatment_name] = str(file_path)
        
        # Print summary for this treatment
        original_cols = len(df.columns)
        new_cols = len(treatment_df.columns) - original_cols
        print(f"✓ {treatment_name}: Added {new_cols} columns, saved to {file_path}")
    
    return output_files 