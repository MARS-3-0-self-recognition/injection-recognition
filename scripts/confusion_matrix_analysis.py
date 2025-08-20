#!/usr/bin/env python3
"""
Simple Confusion Matrix Analysis

This script runs the exact analysis from the last cell of summarizing_results_conf_mat.ipynb
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Fix working directory and Python path
cwd = os.getcwd()
if cwd.endswith("scripts"):
    project_root = os.path.dirname(cwd)
    os.chdir(project_root)
else:
    project_root = cwd

if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"Working directory: {os.getcwd()}")

def load_and_process_data(csv_file_path):
    """Load and process the data from the provided CSV file."""
    
    if not os.path.exists(csv_file_path):
        print(f"CSV file not found: {csv_file_path}")
        return None
    
    # Load data
    df = pd.read_csv(csv_file_path)
    print(f"Loaded data from {csv_file_path}: {df.shape}")
    
    # The CSV files already have processed columns, so we just need to clean up the data
    processed_df = df.copy()
    
    # Ensure we have the required columns
    required_cols = ['eval_id', 'model', 'treatment_type', 'treatment_strength', 'injection_length']
    missing_cols = [col for col in required_cols if col not in processed_df.columns]
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
    
    print(f"Processed data: {processed_df.shape}")
    if 'injection_length' in processed_df.columns:
        print(f"Unique injection lengths: {sorted(processed_df['injection_length'].unique())}")
    if 'treatment_strength' in processed_df.columns:
        print(f"Unique treatment strengths: {sorted(processed_df['treatment_strength'].unique())}")
    if 'treatment_type' in processed_df.columns:
        print(f"Unique treatment types: {sorted(processed_df['treatment_type'].unique())}")
    
    return processed_df

def get_confusion_matrix_columns(df):
    """Get the confusion matrix score columns (the CM_ columns) and their corresponding std columns."""
    # Look for CM_ columns in the CSV files
    conf_matrix_avg_cols = [col for col in df.columns if col.startswith('CM_') and col.endswith('_avg')]
    conf_matrix_std_cols = [col for col in df.columns if col.startswith('CM_') and col.endswith('_std')]
    
    # Sort to ensure consistent order: prefill, consistent, inconsistent for both in_quote and out_quote
    expected_avg_order = [
        'CM_in_quote_prefill_avg',
        'CM_in_quote_consistent_avg', 
        'CM_in_quote_inconsistent_avg',
        'CM_out_quote_prefill_avg',
        'CM_out_quote_consistent_avg',
        'CM_out_quote_inconsistent_avg'
    ]
    
    expected_std_order = [
        'CM_in_quote_prefill_std',
        'CM_in_quote_consistent_std', 
        'CM_in_quote_inconsistent_std',
        'CM_out_quote_prefill_std',
        'CM_out_quote_consistent_std',
        'CM_out_quote_inconsistent_std'
    ]
    
    # Return in expected order if all columns exist
    if all(col in conf_matrix_avg_cols for col in expected_avg_order):
        avg_cols = expected_avg_order
    else:
        print(f"Available CM avg columns: {sorted(conf_matrix_avg_cols)}")
        avg_cols = sorted(conf_matrix_avg_cols)
    
    if all(col in conf_matrix_std_cols for col in expected_std_order):
        std_cols = expected_std_order
    else:
        print(f"Available CM std columns: {sorted(conf_matrix_std_cols)}")
        std_cols = sorted(conf_matrix_std_cols)
    
    return avg_cols, std_cols

def define_conf_mats(df, indices):
    """Build a list of dictionaries containing confusion matrices and metadata for given row indices."""
    # Get confusion matrix columns
    conf_matrix_avg_cols, conf_matrix_std_cols = get_confusion_matrix_columns(df)
    print(f"Found confusion matrix avg columns: {len(conf_matrix_avg_cols)}")
    print("Avg Columns:", conf_matrix_avg_cols)
    print(f"Found confusion matrix std columns: {len(conf_matrix_std_cols)}")
    print("Std Columns:", conf_matrix_std_cols)
    
    if len(conf_matrix_avg_cols) != 6 or len(conf_matrix_std_cols) != 6:
        print(f"Expected 6 avg and 6 std confusion matrix columns, found {len(conf_matrix_avg_cols)} avg and {len(conf_matrix_std_cols)} std")
        return None

    confusion_matrix_2x3_dict = []

    for i, idx in enumerate(indices):
        if idx < len(df):
            row = df.iloc[idx]
            print(f"\n--- Row {i+1} (Index {idx}) ---")
            print(f"Model: {row['model']}")
            print(f"Treatment Type: {row.get('treatment_type', 'N/A')}")
            print(f"Treatment Strength: {row.get('treatment_strength', 'N/A')}")
            print(f"Injection Length: {row.get('injection_length', 'N/A')}")
            
            # Create confusion matrix for this single row using CM_ columns
            # Expected order: prefill, consistent, inconsistent for both in_quote and out_quote
            confusion_matrix_2x3 = pd.DataFrame({
                'Quoted': [
                    row[conf_matrix_avg_cols[0]],  # CM_in_quote_prefill_avg
                    row[conf_matrix_avg_cols[1]],  # CM_in_quote_consistent_avg
                    row[conf_matrix_avg_cols[2]],  # CM_in_quote_inconsistent_avg
                ],
                'Not Quoted': [
                    row[conf_matrix_avg_cols[3]],  # CM_out_quote_prefill_avg
                    row[conf_matrix_avg_cols[4]],  # CM_out_quote_consistent_avg
                    row[conf_matrix_avg_cols[5]],  # CM_out_quote_inconsistent_avg
                ]
            }, index=['Prefill', 'Consistent', 'Inconsistent'])
            
            # Create corresponding standard deviation matrix
            confusion_matrix_std_2x3 = pd.DataFrame({
                'Quoted': [
                    row[conf_matrix_std_cols[0]],  # CM_in_quote_prefill_std
                    row[conf_matrix_std_cols[1]],  # CM_in_quote_consistent_std
                    row[conf_matrix_std_cols[2]],  # CM_in_quote_inconsistent_std
                ],
                'Not Quoted': [
                    row[conf_matrix_std_cols[3]],  # CM_out_quote_prefill_std
                    row[conf_matrix_std_cols[4]],  # CM_out_quote_consistent_std
                    row[conf_matrix_std_cols[5]],  # CM_out_quote_inconsistent_std
                ]
            }, index=['Prefill', 'Consistent', 'Inconsistent'])
            
            print(f"Confusion Matrix Values (Avg ± Std):")
            print(confusion_matrix_2x3)
            print(f"Standard Deviations:")
            print(confusion_matrix_std_2x3)
            
            # Create dictionary entry for this row
            row_dict = {
                "eval_id": row["eval_id"],
                "model": row["model"],
                "treatment_type": row.get("treatment_type", "Unknown"),
                "treatment_strength": row.get("treatment_strength", "Unknown"),
                "injection_length": row.get("injection_length", "Unknown"),
                "confusion_matrix_2x3": confusion_matrix_2x3,
                "confusion_matrix_std_2x3": confusion_matrix_std_2x3,
                "row_index": idx
            }
            
            confusion_matrix_2x3_dict.append(row_dict)
    
    return confusion_matrix_2x3_dict

def plot_confusion_matrix(confusion_matrix_2x3_dict, output_dir, csv_filename):
    """Plot all confusion matrices from the list of dictionaries and save to files."""
    if not confusion_matrix_2x3_dict:
        print("No confusion matrices to plot")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for i, row_dict in enumerate(confusion_matrix_2x3_dict):
        confusion_matrix_2x3 = row_dict["confusion_matrix_2x3"]
        confusion_matrix_std_2x3 = row_dict["confusion_matrix_std_2x3"]
        
        # Create the plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        
        # Create custom annotations that include both avg and std
        annotations = []
        for row_idx in range(confusion_matrix_2x3.shape[0]):
            row_annotations = []
            for col_idx in range(confusion_matrix_2x3.shape[1]):
                avg_val = confusion_matrix_2x3.iloc[row_idx, col_idx]
                std_val = confusion_matrix_std_2x3.iloc[row_idx, col_idx]
                annotation = f"{avg_val:.3f}\n±{std_val:.3f}"
                row_annotations.append(annotation)
            annotations.append(row_annotations)
        
        # Create heatmap with custom annotations
        sns.heatmap(confusion_matrix_2x3, annot=annotations, fmt='', cmap='Blues', 
                    ax=ax, cbar_kws={'label': 'Score (Mean ± Std)'})
        
        # Clean model name for filename
        model_name = row_dict["model"].split('/')[-1] if '/' in row_dict["model"] else row_dict["model"]
        
        ax.set_title(f'Row {i+1} - {model_name}\n{row_dict["treatment_type"]} {row_dict["treatment_strength"]} (IL{row_dict["injection_length"]})\nValues shown as Mean ± Standard Deviation', 
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Quote Status', fontweight='bold')
        ax.set_ylabel('Word Category', fontweight='bold')
        
        plt.tight_layout()
        
        # Create a shorter filename to avoid Windows path length limits
        # Extract key parts for naming
        csv_short = csv_filename.split('_')[-3:]  # Get last 3 parts like ['caps', 'IL20', '']
        csv_short = '_'.join([part for part in csv_short if part])  # Remove empty parts
        
        filename = f"row_{i+1}_{model_name}_{row_dict['treatment_type']}_{row_dict['treatment_strength']}.png"
        # Clean filename of any problematic characters
        filename = filename.replace('/', '_').replace('\\', '_').replace(':', '_')
        
        filepath = os.path.join(output_dir, filename)
        
        # Normalize path separators for Windows
        filepath = os.path.normpath(filepath)
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {filepath}")
        
        plt.close(fig)


def process_csv_file(csv_file_path, base_output_dir):
    """Process a single CSV file and generate plots."""
    print(f"\n=== Processing {csv_file_path} ===")
    
    # Load and process data
    processed_df = load_and_process_data(csv_file_path)
    if processed_df is None:
        return
    
    # Extract csv filename without extension for naming
    csv_filename = os.path.basename(csv_file_path).replace('.csv', '')
    
    # Create short directory name based on csv filename
    if 'rows_caps_IL20' in csv_filename:
        dir_name = 'rows_caps_IL20'
    elif 'rows_typo_IL20' in csv_filename:
        dir_name = 'rows_typo_IL20'
    elif 'columns_caps_IL20' in csv_filename:
        dir_name = 'columns_caps_IL20'
    elif 'columns_typo_IL20' in csv_filename:
        dir_name = 'columns_typo_IL20'
    else:
        dir_name = csv_filename
    
    # Create output directory
    output_dir = os.path.join(base_output_dir, dir_name)
    
    # Get all rows (excluding header, so subtract 1 from length)
    num_rows = len(processed_df)
    row_indices = list(range(num_rows))
    
    print(f"Processing {num_rows} rows from {csv_filename}")
    
    # Generate confusion matrices for all rows
    confusion_matrices = define_conf_mats(processed_df, row_indices)
    
    if confusion_matrices:
        plot_confusion_matrix(confusion_matrices, output_dir, csv_filename)
        print(f"Generated {len(confusion_matrices)} plots in {output_dir}")
    

def debug_single_chart():
    """Debug function to save just a single chart for testing."""
    print("=== DEBUG: Single Chart Mode ===")
    
    # Use the first CSV file and first row for testing
    base_path = "scripts/data/wikihow_summary_injection_conf_mat/treatment"
    csv_file = "avg_evals_df_per_sample_norm_confusion_matrix_rows_caps_IL20.csv"
    csv_path = os.path.join(base_path, csv_file)
    
    if not os.path.exists(csv_path):
        print(f"Debug CSV file not found: {csv_path}")
        return
    
    # Load and process data
    processed_df = load_and_process_data(csv_path)
    if processed_df is None:
        return
    
    # Create output directory for debug
    debug_output_dir = os.path.join(base_path, "debug_single_chart")
    
    # Generate confusion matrix for just the first row
    confusion_matrices = define_conf_mats(processed_df, [0])  # Only first row
    
    if confusion_matrices:
        csv_filename = os.path.basename(csv_path).replace('.csv', '')
        plot_confusion_matrix(confusion_matrices, debug_output_dir, csv_filename)
        print(f"DEBUG: Generated 1 plot in {debug_output_dir}")
    else:
        print("DEBUG: No confusion matrices generated")


def main():
    """Main function to run the analysis."""
    print("=== Simple Confusion Matrix Analysis ===")
    
    # Check for debug mode - uncomment the next line to enable debug mode
    # debug_single_chart(); return
    
    # Define the CSV files to process
    base_path = "scripts/data/wikihow_summary_injection_conf_mat/treatment"
    '''csv_files = [
        "avg_evals_df_per_sample_norm_confusion_matrix_rows_caps_IL20.csv",
        "avg_evals_df_per_sample_norm_confusion_matrix_rows_typo_IL20.csv", 
        "avg_evals_df_per_sample_norm_confusion_matrix_columns_caps_IL20.csv",
        "avg_evals_df_per_sample_norm_confusion_matrix_columns_typo_IL20.csv"
    ]'''
    csv_files = [
        "avg_evals_df_per_sample_norm_confusion_matrix_total_caps_IL20.csv",
        "avg_evals_df_per_sample_norm_confusion_matrix_total_typo_IL20.csv"
    ]
    
    # Output directory (same as input directory)
    base_output_dir = base_path
    
    total_plots = 0
    
    # Process each CSV file
    for csv_file in csv_files:
        csv_path = os.path.join(base_path, csv_file)
        if os.path.exists(csv_path):
            process_csv_file(csv_path, base_output_dir)
            # Count plots (approximately 3 per file based on visible data)
            total_plots += 3  # This will be updated dynamically by the actual function
        else:
            print(f"Warning: CSV file not found: {csv_path}")
    
    print(f"\n=== Analysis Complete ===")
    print(f"Processed {len(csv_files)} CSV files")
    print(f"Generated plots saved in subdirectories of: {base_output_dir}")


if __name__ == "__main__":
    main()
