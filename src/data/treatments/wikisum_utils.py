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
import random

# Add the current directory to the Python path to import our modules
import sys
sys.path.append(str(Path(__file__).parent))

from string_modifier import randomly_capitalize_string
from wiki_article_processor import WikiArticleProcessor

# Global dataset cache to avoid reloading
# Now supports cache keys like 'huggingface_all', 'huggingface_train_validation', 'local_all', etc.
_dataset_cache = {}


def _get_cached_dataset(use_huggingface: bool = True, splits: Optional[List[str]] = None):
    """
    Get dataset from cache or load it if not cached.
    
    Args:
        use_huggingface (bool): Whether to use HuggingFace datasets
        splits (Optional[List[str]]): Which splits to include ['train', 'validation', 'test'].
                                     If None, includes all splits.
        
    Returns:
        List[dict]: Cached or newly loaded articles
    """
    # Create cache key that includes splits info
    if splits is None:
        splits_key = "all"
    else:
        splits_key = "_".join(sorted(splits))
    
    cache_key = f"{'huggingface' if use_huggingface else 'local'}_{splits_key}"
    
    if cache_key not in _dataset_cache or _dataset_cache[cache_key] is None:
        print(f"Loading WikiSum dataset ({'HuggingFace' if use_huggingface else 'local'}) with splits {splits or 'all'} - this will be cached for future use...")
        processor = WikiArticleProcessor()
        
        if use_huggingface:
            all_articles = processor.load_wikisum_dataset(use_huggingface=True)
            
            # Filter by splits if specified
            if splits is not None:
                valid_splits = ['train', 'validation', 'test']
                invalid_splits = [s for s in splits if s not in valid_splits]
                if invalid_splits:
                    print(f"Warning: Invalid splits specified: {invalid_splits}. Valid splits are: {valid_splits}")
                
                valid_requested_splits = [s for s in splits if s in valid_splits]
                if not valid_requested_splits:
                    raise ValueError(f"No valid splits specified. Valid splits are: {valid_splits}")
                
                articles = [article for article in all_articles 
                           if article.get('split') in valid_requested_splits]
                print(f"Filtered to {len(articles)} articles from splits: {valid_requested_splits}")
            else:
                articles = all_articles
        else:
            # Local datasets typically don't have split information
            articles = processor.load_wikisum_dataset(use_huggingface=False)
            if splits is not None:
                print("Warning: Split filtering is not supported for local datasets. Loading all data.")
            
        _dataset_cache[cache_key] = articles
        print(f"Dataset cached! Future calls will be much faster.")
    else:
        print(f"Using cached dataset ({len(_dataset_cache[cache_key])} articles)")
        
    return _dataset_cache[cache_key]


def _filter_columns(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Filter DataFrame to keep only specified columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (Optional[List[str]]): List of column names to keep. If None, keep all columns.
        
    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    if columns is None:
        return df
    
    available_columns = df.columns.tolist()
    missing_columns = [col for col in columns if col not in available_columns]
    if missing_columns:
        print(f"Warning: Columns not found in data: {missing_columns}")
    
    valid_columns = [col for col in columns if col in available_columns]
    if valid_columns:
        return df[valid_columns]
    else:
        print("Warning: No valid columns specified, keeping all columns")
        return df


def _generate_filename(start_idx: Optional[int] = None,
                      end_idx: Optional[int] = None,
                      indices: Optional[List[int]] = None,
                      n_random: Optional[int] = None,
                      seed: Optional[int] = None) -> str:
    """
    Generate filename based on parameters.
    
    Args:
        start_idx (Optional[int]): Starting index for range-based selection
        end_idx (Optional[int]): Ending index for range-based selection
        indices (Optional[List[int]]): List of specific indices
        n_random (Optional[int]): Number of random articles
        seed (Optional[int]): Random seed used
        
    Returns:
        str: Generated filename
    """
    if indices is not None:
        # Specific indices mode
        unique_indices = sorted(list(set(indices)))
        n_count = len(unique_indices)
        seed_str = f"_seed{seed}" if seed is not None else ""
        return f"wikisum_n{n_count}{seed_str}.csv"
    elif n_random is not None:
        # Random mode
        seed_str = f"_seed{seed}" if seed is not None else ""
        return f"wikisum_random_n{n_random}{seed_str}.csv"
    elif start_idx is not None and end_idx is not None:
        # Range-based mode
        return f"wikisum_{start_idx}_{end_idx}.csv"
    else:
        raise ValueError("Cannot generate filename: insufficient parameters")


def _save_to_csv(df: pd.DataFrame, 
                save_path: Optional[Union[str, Path]], 
                start_idx: Optional[int] = None,
                end_idx: Optional[int] = None,
                indices: Optional[List[int]] = None,
                n_random: Optional[int] = None,
                seed: Optional[int] = None) -> None:
    """
    Save DataFrame to CSV if requested.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        save_path (Optional[Union[str, Path]]): 
            - None: don't save
            - Directory path: save with auto-generated filename
            - File path: save to this exact path
        start_idx, end_idx, indices, n_random, seed: Parameters for filename generation
    """
    if save_path is None:
        return
    
    file_path = Path(save_path)
    
    # If save_path is a directory, generate filename automatically
    if file_path.is_dir() or (not file_path.suffix and not file_path.exists()):
        # It's a directory or looks like a directory (no file extension)
        filename = _generate_filename(start_idx, end_idx, indices, n_random, seed)
        file_path = file_path / filename
    
    # Ensure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving to CSV: {file_path}")
    df.to_csv(file_path, index=False)


def _get_cached_csv(file_path: Path, columns: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    """
    Load DataFrame from cached CSV file with optional column filtering.
    
    Args:
        file_path (Path): Path to cached CSV file
        columns (Optional[List[str]]): Columns to keep
        
    Returns:
        Optional[pd.DataFrame]: Loaded DataFrame if file exists, None otherwise
    """
    if not file_path.exists():
        return None
    
    print(f"Loading existing CSV: {file_path}")
    df = pd.read_csv(file_path)
    return _filter_columns(df, columns)


def get_WikiSum(start_idx: Optional[int] = None, 
                end_idx: Optional[int] = None, 
                indices: Optional[List[int]] = None,
                use_huggingface: bool = True,
                splits: Optional[List[str]] = None,
                save_path: Optional[Union[str, Path]] = None,
                columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Get WikiSum articles by index range, specific indices, or return as DataFrame.
    
    Args:
        start_idx (Optional[int]): Starting index (inclusive) - for range-based selection
        end_idx (Optional[int]): Ending index (exclusive) - for range-based selection  
        indices (Optional[List[int]]): List of specific article indices to grab
        use_huggingface (bool): If True, load from Hugging Face datasets library
        splits (Optional[List[str]]): Which dataset splits to include ['train', 'validation', 'test'].
                                     If None, includes all splits. Only works with HuggingFace datasets.
        save_path (Optional[Union[str, Path]]): Path to save CSV. If None, don't save.
        columns (Optional[List[str]]): List of column names to keep. If None, keep all columns.
        
    Returns:
        pd.DataFrame: DataFrame containing the requested articles
        
    Raises:
        ValueError: If invalid parameters or no articles found
    """
    # Validate input parameters
    if indices is not None:
        # Specific indices mode
        if not isinstance(indices, list) or len(indices) == 0:
            raise ValueError("indices must be a non-empty list of integers")
        if not all(isinstance(i, int) and i >= 0 for i in indices):
            raise ValueError("All indices must be non-negative integers")
        
        # Remove duplicates and sort for consistent caching
        unique_indices = sorted(list(set(indices)))
        indices_str = "_".join(map(str, unique_indices))
        filename = f"wikisum_indices_{indices_str}.csv"
        
    elif start_idx is not None and end_idx is not None:
        # Range-based mode (original functionality)
        if start_idx < 0 or end_idx <= start_idx:
            raise ValueError(f"Invalid range: start_idx={start_idx}, end_idx={end_idx}")
        
        filename = f"wikisum_{start_idx}_{end_idx}.csv"
        
    else:
        raise ValueError("Must provide either (start_idx, end_idx) or indices")
    
    # Check for cached file (only when not saving or save_path matches cache)
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    cache_path = output_dir / filename
    
    if cache_path.exists() and (save_path is None or Path(save_path) == cache_path):
        df = _get_cached_csv(cache_path, columns)
        if df is not None:
            return df
    
    # Load from WikiSum dataset
    if indices is not None:
        print(f"Loading WikiSum articles with indices: {unique_indices}")
    else:
        end_idx_print = end_idx - 1 if end_idx is not None else 'None'
        print(f"Loading WikiSum articles {start_idx} to {end_idx_print}...")
    
    articles = _get_cached_dataset(use_huggingface, splits)
    
    if not articles:
        raise ValueError("No articles found in WikiSum dataset")
    
    # Select articles based on mode
    if indices is not None:
        # Specific indices mode
        max_idx = len(articles) - 1
        valid_indices = [i for i in unique_indices if i <= max_idx]
        
        if len(valid_indices) != len(unique_indices):
            invalid_indices = [i for i in unique_indices if i > max_idx]
            print(f"Warning: Some indices exceed available articles ({len(articles)}). Skipping: {invalid_indices}")
        
        if not valid_indices:
            raise ValueError(f"No valid indices found. Max available index: {max_idx}")
        
        selected_articles = [articles[i] for i in valid_indices]
        
    else:
        # Range-based mode
        if end_idx is not None and end_idx > len(articles):
            print(f"Warning: end_idx {end_idx} exceeds available articles ({len(articles)}). Using {len(articles)} instead.")
            end_idx = len(articles)
        
        selected_articles = articles[start_idx:end_idx]
    
    if not selected_articles:
        raise ValueError(f"No articles found for the specified selection")
    
    # Convert to DataFrame
    df = pd.DataFrame(selected_articles)
    
    # Filter columns if specified
    df = _filter_columns(df, columns)
    
    # Save to CSV if requested
    _save_to_csv(df, save_path, start_idx, end_idx, indices)
    
    print(f"Loaded {len(df)} articles")
    return df


def get_WikiSum_random(n: int, 
                      use_huggingface: bool = True,
                      seed: Optional[int] = None,
                      splits: Optional[List[str]] = None,
                      save_path: Optional[Union[str, Path]] = None,
                      columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Get N random WikiSum articles efficiently.
    
    Args:
        n (int): Number of random articles to grab
        use_huggingface (bool): If True, load from Hugging Face datasets library
        seed (Optional[int]): Random seed for reproducible results
        splits (Optional[List[str]]): Which dataset splits to include ['train', 'validation', 'test'].
                                     If None, includes all splits. Only works with HuggingFace datasets.
        save_path (Optional[Union[str, Path]]): Path to save CSV. If None, don't save.
        columns (Optional[List[str]]): List of column names to keep. If None, keep all columns.
        
    Returns:
        pd.DataFrame: DataFrame containing the requested random articles
        
    Raises:
        ValueError: If invalid parameters or no articles found
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")
    
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
    
    # Load dataset once and sample from it
    print(f"Requesting {n} random articles...")
    articles = _get_cached_dataset(use_huggingface, splits)
    
    if not articles:
        raise ValueError("No articles found in WikiSum dataset")
    
    max_articles = len(articles)
    
    if n > max_articles:
        print(f"Warning: Requested {n} articles but only {max_articles} available. Using {max_articles} instead.")
        n = max_articles
    
    # Sample random indices and get those articles directly
    random_indices = random.sample(range(max_articles), n)
    selected_articles = [articles[i] for i in random_indices]
    
    print(f"Selected random indices: {random_indices}")
    
    # Convert to DataFrame
    df = pd.DataFrame(selected_articles)
    
    # Filter columns if specified
    df = _filter_columns(df, columns)
    
    # Save to CSV if requested
    _save_to_csv(df, save_path, n_random=n, seed=seed)
    
    print(f"Loaded {len(df)} random articles")
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


def show_available_splits(use_huggingface: bool = True) -> dict:
    """
    Show available dataset splits and their sizes.
    
    Args:
        use_huggingface (bool): If True, check HuggingFace datasets, otherwise local datasets
        
    Returns:
        dict: Dictionary with split names as keys and article counts as values
    """
    if not use_huggingface:
        print("Split information is only available for HuggingFace datasets.")
        return {}
    
    # Load all articles to check splits
    articles = _get_cached_dataset(use_huggingface=True, splits=None)
    
    # Count articles by split
    split_counts = {}
    for article in articles:
        split_name = article.get('split', 'unknown')
        split_counts[split_name] = split_counts.get(split_name, 0) + 1
    
    print("Available dataset splits:")
    print("-" * 30)
    for split_name, count in sorted(split_counts.items()):
        print(f"{split_name:>12}: {count:>6} articles")
    print("-" * 30)
    print(f"{'Total':>12}: {sum(split_counts.values()):>6} articles")
    
    return split_counts 