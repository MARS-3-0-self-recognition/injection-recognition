# Fix working directory and Python path to find src module from scripts directory
import sys
import os
import altair as alt
import pandas as pd
from inspect_ai.log import list_eval_logs, read_eval_log, write_eval_log, EvalLogInfo, EvalLog
from inspect_ai.analysis import evals_df, samples_df
from src.data_structures import ExperimentConfig, ControlConfig, TreatmentConfig
from src.inspect_helpers.tasks import injection_consistency_and_recognition
from src.inspect_helpers.datasets import ROW_INDEX_KEY
from src.inspect_helpers.scorers import answer_match, which_treatment_mgf
from src.inspect_helpers.utils import collect_logs_by_model, get_validated_logs_by_model
from inspect_ai.log import EvalLog, list_eval_logs, read_eval_log
from inspect_ai.model import (
    Model,
    ModelAPI,
    GenerateConfig,
    anthropic,
    ollama,
    get_model,
)
from inspect_ai import eval, eval_async
import pandas as pd
import os
import numpy as np
from inspect_ai.analysis import evals_df


def clean_confusion_matrix_header(key):
    """
    Clean up confusion matrix column headers by simplifying the naming convention.
    
    Maps headers like:
    - 'in_task2_quote__prefill_words' -> 'in_quote_prefill'
    - 'not_in_task2_quote__consistent_words' -> 'out_quote_consistent'
    - 'not_in_task2_quote__inconsistent_words' -> 'out_quote_inconsistent'
    """
    # Check for 'not_in_task2_quote' first (more specific)
    if 'not_in_task2_quote' in key:
        if 'prefill_words' in key:
            return 'out_quote_prefill'
        elif '_inconsistent_words' in key:  # Check inconsistent first (more specific)
            return 'out_quote_inconsistent'
        elif '_consistent_words' in key:
            return 'out_quote_consistent'
    elif 'in_task2_quote' in key:
        if 'prefill_words' in key:
            return 'in_quote_prefill'
        elif '_inconsistent_words' in key:  # Check inconsistent first (more specific)
            return 'in_quote_inconsistent'
        elif '_consistent_words' in key:
            return 'in_quote_consistent'
    
    # Return original key if no pattern matches
    return key


##
def extract_treatment_data_with_conf_matrix(treatment_log_dir):
    '''Extract treatment data with a row for every sample instead of aggregating.'''
    
    treatment_logs = list_eval_logs(treatment_log_dir, filter=lambda log: log.status == "success")
    
    if not treatment_logs:
        print("No successful treatment logs found")
        return None
    
    # Get basic DataFrame first to extract metadata fields
    print("Creating basic DataFrame from treatment logs...")
    treatment_df = evals_df(treatment_logs)
    print(f"Basic treatment DF shape: {treatment_df.shape}")
    
    # Extract sample-level data
    print("Extracting sample-level data...")
    samples_data = []
    
    for log_info in treatment_logs:
        log_path = log_info.name.replace('file://', '')
        try:
            log = read_eval_log(log_path)
            
            # Get the corresponding row from the basic DataFrame for metadata
            log_row = treatment_df[treatment_df['eval_id'] == log.eval.eval_id].iloc[0]
            
            # Extract data from each sample
            if hasattr(log, 'samples') and log.samples:
                for sample_idx, sample in enumerate(log.samples):
                    sample_data = {}
                    
                    # Add only the specific metadata fields requested
                    sample_data['eval_id'] = log_row['eval_id']
                    sample_data['task_id'] = log_row['task_id']
                    sample_data['model'] = log_row['model']
                    sample_data['dataset_samples'] = log_row['dataset_samples']
                    sample_data['score_answer_match_accuracy'] = log_row['score_answer_match_accuracy']
                    sample_data['score_answer_match_stderr'] = log_row['score_answer_match_stderr']
                    sample_data['score_which_treatment_mgf_accuracy'] = log_row['score_which_treatment_mgf_accuracy']
                    sample_data['score_which_treatment_mgf_stderr'] = log_row['score_which_treatment_mgf_stderr']
                    
                    # Parse task_arg_treatment_col into two columns
                    treatment_col = log_row['task_arg_treatment_col']
                    if pd.isna(treatment_col):
                        sample_data['injection_length'] = None
                        sample_data['treatment_strength'] = None
                    else:
                        # Split by underscore and extract parts
                        parts = str(treatment_col).split('_')
                        if len(parts) >= 2:
                            # Remove "IL" prefix from injection length
                            injection_length = parts[0]
                            if injection_length.startswith('IL'):
                                injection_length = injection_length[2:]  # Remove first 2 characters
                            sample_data['injection_length'] = injection_length
                            if parts[1] == 'medium':
                                sample_data['treatment_strength'] = "S0"
                            elif parts[1] == 'heavy':
                                sample_data['treatment_strength'] = "S4"
                            else:
                                sample_data['treatment_strength'] = parts[1]
                        else:
                            # Remove "IL" prefix if present
                            injection_length = str(treatment_col)
                            if injection_length.startswith('IL'):
                                injection_length = injection_length[2:]
                            sample_data['injection_length'] = injection_length
                            sample_data['treatment_strength'] = None
                    
                    # Add sample-specific identifiers
                    #sample_data['sample_idx'] = sample_idx
                    #sample_data['sample_id'] = getattr(sample, 'id', f'sample_{sample_idx}')
                    
                    # Extract all available scores from the sample
                    if hasattr(sample, 'scores'):
                        # Extract answer_match score
                        if 'answer_match' in sample.scores:
                            answer_match_score = sample.scores['answer_match']
                            if hasattr(answer_match_score, 'value'):
                                raw_value = answer_match_score.value
                            else:
                                raw_value = answer_match_score
                            
                            # Convert I/C to numeric scores
                            if raw_value == "I":
                                sample_data['answer_match'] = 0
                            elif raw_value == "C":
                                sample_data['answer_match'] = 1
                            else:
                                sample_data['answer_match'] = raw_value
                        
                        # Extract which_treatment_mgf score
                        if 'which_treatment_mgf' in sample.scores:
                            which_treatment_score = sample.scores['which_treatment_mgf']
                            if hasattr(which_treatment_score, 'value'):
                                raw_value = which_treatment_score.value
                            else:
                                raw_value = which_treatment_score
                            
                            # Convert I/C to numeric scores
                            if raw_value == "I":
                                sample_data['which_treatment_mgf'] = 0
                            elif raw_value == "C":
                                sample_data['which_treatment_mgf'] = 1
                            else:
                                sample_data['which_treatment_mgf'] = raw_value
                        
                        # Extract confusion matrix scores if available
                        if 'consistency_recognition_conf_matrix' in sample.scores:
                            conf_score = sample.scores['consistency_recognition_conf_matrix']
                            if hasattr(conf_score, 'value') and isinstance(conf_score.value, dict):
                                for key, value in conf_score.value.items():
                                    # Clean up confusion matrix column headers
                                    clean_key = clean_confusion_matrix_header(key)
                                    sample_data[f'CM_{clean_key}'] = value
                        
                        # Extract any other scores that might be present
                        for score_name, score_value in sample.scores.items():
                            if score_name not in ['answer_match', 'which_treatment_mgf', 'consistency_recognition_conf_matrix']:
                                if hasattr(score_value, 'value'):
                                    sample_data[f'score_{score_name}'] = score_value.value
                                else:
                                    sample_data[f'score_{score_name}'] = score_value
                    
                    # Add treatment type based on sample_metadata_csv_file
                    if hasattr(sample, 'metadata') and 'csv_file' in sample.metadata:
                        csv_file = sample.metadata['csv_file']
                        if 'dataset_typo_rates_injected.csv' in csv_file:
                            sample_data['treatment_type'] = 'typo'
                        elif 'dataset_capitalization_rates_injected.csv' in csv_file:
                            sample_data['treatment_type'] = 'capitalization'
                        else:
                            sample_data['treatment_type'] = 'unknown'
                    else:
                        sample_data['treatment_type'] = None
                    
                    samples_data.append(sample_data)
            
        except Exception as e:
            print(f"Error processing log {log_path}: {e}")
            continue
    
    # Convert to DataFrame
    if samples_data:
        samples_df = pd.DataFrame(samples_data)
        
        # Reorder columns to ensure treatment_type comes after injection_length
        column_order = [
            'eval_id', 'task_id', 'model', 'dataset_samples',
            'score_answer_match_accuracy', 'score_answer_match_stderr',
            'score_which_treatment_mgf_accuracy', 'score_which_treatment_mgf_stderr',
            'injection_length', 'treatment_type', 'treatment_strength',
            'answer_match', 'which_treatment_mgf'
        ]
        
        # Add any confusion matrix columns that exist
        for col in samples_df.columns:
            if col.startswith('CM_') and col not in column_order:
                column_order.append(col)
        
        # Add any other score columns that exist
        for col in samples_df.columns:
            if col.startswith('score_') and col not in column_order:
                column_order.append(col)
        
        # Reorder the DataFrame
        samples_df = samples_df[column_order]
        
        print(f"Sample-level DF shape: {samples_df.shape}")
        return samples_df
    else:
        print("No sample data found")
        return treatment_df

def process_per_sample(df):
    eval_ids = df['eval_id'].unique()
    new_df = pd.DataFrame()
    for eval_id in eval_ids:
        eval_df = df[df['eval_id'] == eval_id]
        sample_data = eval_df.loc[:, 'eval_id':'treatment_strength'].iloc[0]
        category_data = eval_df.loc[:, 'answer_match':'which_treatment_mgf']
        CM_data = eval_df.loc[:, 'CM_in_quote_prefill':'CM_out_quote_inconsistent']
        category_data_row_sum = category_data.sum(axis=0)
        CM_data_row_sum = CM_data.sum(axis=0)
        category_data_row_avg = category_data.mean(axis=0)
        CM_data_row_avg = CM_data.mean(axis=0)
        category_data_row_std = category_data.std(axis=0)
        CM_data_row_std = CM_data.std(axis=0)
        new_row = pd.concat([sample_data, category_data_row_sum, CM_data_row_sum, category_data_row_avg, CM_data_row_avg, category_data_row_std, CM_data_row_std], axis=0)
        new_df = pd.concat([new_df, new_row], axis=1)
    return new_df

def process_avg_per_sample(df):
    eval_ids = df['eval_id'].unique()
    new_df = pd.DataFrame()
    for eval_id in eval_ids:
        eval_df = df[df['eval_id'] == eval_id]
        sample_data = eval_df.loc[:, 'eval_id':'treatment_strength'].iloc[0]
        category_data = eval_df.loc[:, 'answer_match':'which_treatment_mgf']
        CM_data = eval_df.loc[:, 'CM_in_quote_prefill':'CM_out_quote_inconsistent']
        
        # Calculate averages and standard deviations
        category_data_row_avg = category_data.mean(axis=0)
        CM_data_row_avg = CM_data.mean(axis=0)
        category_data_row_std = category_data.std(axis=0)
        CM_data_row_std = CM_data.std(axis=0)
        
        # Rename columns to append _avg and _std
        category_data_row_avg.index = [col + '_avg' for col in category_data_row_avg.index]
        CM_data_row_avg.index = [col + '_avg' for col in CM_data_row_avg.index]
        category_data_row_std.index = [col + '_std' for col in category_data_row_std.index]
        CM_data_row_std.index = [col + '_std' for col in CM_data_row_std.index]
        
        # Combine all data: original sample data + averaged category data + averaged CM data + std category data + std CM data
        new_row = pd.concat([sample_data, category_data_row_avg, category_data_row_std, CM_data_row_avg, CM_data_row_std], axis=0)
        new_df = pd.concat([new_df, new_row], axis=1)
    
    # Transpose the DataFrame to get proper column structure
    new_df = new_df.T
    
    return new_df

def process_avg_per_sample_norm_confusion_matrix_rows(df):
    eval_ids = df['eval_id'].unique()
    new_df = pd.DataFrame()
    for eval_id in eval_ids:
        eval_df = df[df['eval_id'] == eval_id]
        sample_data = eval_df.loc[:, 'eval_id':'treatment_strength'].iloc[0]
        category_data = eval_df.loc[:, 'answer_match':'which_treatment_mgf']
        CM_data = eval_df.loc[:, 'CM_in_quote_prefill':'CM_out_quote_inconsistent']
        
        # Calculate averages and standard deviations on raw data first
        category_data_row_avg = category_data.mean(axis=0)
        CM_data_row_avg = CM_data.mean(axis=0)
        category_data_row_std = category_data.std(axis=0)
        CM_data_row_std = CM_data.std(axis=0)
        
        # Now normalize the averaged CM data by category (prefill, consistent, inconsistent)
        CM_data_row_avg_normalized = CM_data_row_avg.copy()
        CM_data_row_std_normalized = CM_data_row_std.copy()
        
        # Get columns for each category
        prefill_cols = [col for col in CM_data_row_avg.index if '_prefill' in col]
        consistent_cols = [col for col in CM_data_row_avg.index if '_consistent' in col]
        inconsistent_cols = [col for col in CM_data_row_avg.index if '_inconsistent' in col]
        
        # Calculate totals for each category from averages
        prefill_total = CM_data_row_avg[prefill_cols].sum()
        consistent_total = CM_data_row_avg[consistent_cols].sum()
        inconsistent_total = CM_data_row_avg[inconsistent_cols].sum()
        
        # Normalize averages by category (avoid division by zero)
        if prefill_total > 0:
            CM_data_row_avg_normalized[prefill_cols] = CM_data_row_avg[prefill_cols] / prefill_total
            # Also normalize the standard deviations
            CM_data_row_std_normalized[prefill_cols] = CM_data_row_std[prefill_cols] / prefill_total
        else:
            CM_data_row_avg_normalized[prefill_cols] = 0
            CM_data_row_std_normalized[prefill_cols] = 0
            
        if consistent_total > 0:
            CM_data_row_avg_normalized[consistent_cols] = CM_data_row_avg[consistent_cols] / consistent_total
            # Also normalize the standard deviations
            CM_data_row_std_normalized[consistent_cols] = CM_data_row_std[consistent_cols] / consistent_total
        else:
            CM_data_row_avg_normalized[consistent_cols] = 0
            CM_data_row_std_normalized[consistent_cols] = 0
            
        if inconsistent_total > 0:
            CM_data_row_avg_normalized[inconsistent_cols] = CM_data_row_avg[inconsistent_cols] / inconsistent_total
            # Also normalize the standard deviations
            CM_data_row_std_normalized[inconsistent_cols] = CM_data_row_std[inconsistent_cols] / inconsistent_total
        else:
            CM_data_row_avg_normalized[inconsistent_cols] = 0
            CM_data_row_std_normalized[inconsistent_cols] = 0
        
        # Rename columns to append _avg and _std
        category_data_row_avg.index = [col + '_avg' for col in category_data_row_avg.index]
        CM_data_row_avg_normalized.index = [col + '_avg' for col in CM_data_row_avg_normalized.index]
        category_data_row_std.index = [col + '_std' for col in category_data_row_std.index]
        CM_data_row_std_normalized.index = [col + '_std' for col in CM_data_row_std_normalized.index]
        
        # Combine all data: original sample data + averaged category data + averaged CM data + std category data + std CM data
        new_row = pd.concat([sample_data, category_data_row_avg, category_data_row_std, CM_data_row_avg_normalized, CM_data_row_std_normalized], axis=0)
        new_df = pd.concat([new_df, new_row], axis=1)
    
    # Transpose the DataFrame to get proper column structure
    new_df = new_df.T
    
    return new_df

def process_avg_per_sample_norm_confusion_matrix_columns(df):
    eval_ids = df['eval_id'].unique()
    new_df = pd.DataFrame()
    for eval_id in eval_ids:
        eval_df = df[df['eval_id'] == eval_id]
        sample_data = eval_df.loc[:, 'eval_id':'treatment_strength'].iloc[0]
        category_data = eval_df.loc[:, 'answer_match':'which_treatment_mgf']
        CM_data = eval_df.loc[:, 'CM_in_quote_prefill':'CM_out_quote_inconsistent']
        
        # Calculate averages and standard deviations on raw data first
        category_data_row_avg = category_data.mean(axis=0)
        CM_data_row_avg = CM_data.mean(axis=0)
        category_data_row_std = category_data.std(axis=0)
        CM_data_row_std = CM_data.std(axis=0)
        
        # Now normalize the averaged CM data by category (in_quote, out_quote)
        CM_data_row_avg_normalized = CM_data_row_avg.copy()
        CM_data_row_std_normalized = CM_data_row_std.copy()
        
        # Get columns for each category
        in_quote_cols = [col for col in CM_data_row_avg.index if 'in_quote' in col]
        out_quote_cols = [col for col in CM_data_row_avg.index if 'out_quote' in col]
        
        # Calculate totals for each category from averages
        in_quote_total = CM_data_row_avg[in_quote_cols].sum()
        out_quote_total = CM_data_row_avg[out_quote_cols].sum()
        
        # Normalize averages by category (avoid division by zero)
        if in_quote_total > 0:
            CM_data_row_avg_normalized[in_quote_cols] = CM_data_row_avg[in_quote_cols] / in_quote_total
            # Also normalize the standard deviations
            CM_data_row_std_normalized[in_quote_cols] = CM_data_row_std[in_quote_cols] / in_quote_total
        else:
            CM_data_row_avg_normalized[in_quote_cols] = 0
            CM_data_row_std_normalized[in_quote_cols] = 0
            
        if out_quote_total > 0:
            CM_data_row_avg_normalized[out_quote_cols] = CM_data_row_avg[out_quote_cols] / out_quote_total
            # Also normalize the standard deviations
            CM_data_row_std_normalized[out_quote_cols] = CM_data_row_std[out_quote_cols] / out_quote_total
        else:
            CM_data_row_avg_normalized[out_quote_cols] = 0
            CM_data_row_std_normalized[out_quote_cols] = 0
        
        # Rename columns to append _avg and _std
        category_data_row_avg.index = [col + '_avg' for col in category_data_row_avg.index]
        CM_data_row_avg_normalized.index = [col + '_avg' for col in CM_data_row_avg_normalized.index]
        category_data_row_std.index = [col + '_std' for col in category_data_row_std.index]
        CM_data_row_std_normalized.index = [col + '_std' for col in CM_data_row_std_normalized.index]
        
        # Combine all data: original sample data + averaged category data + averaged CM data + std category data + std CM data
        new_row = pd.concat([sample_data, category_data_row_avg, category_data_row_std, CM_data_row_avg_normalized, CM_data_row_std_normalized], axis=0)
        new_df = pd.concat([new_df, new_row], axis=1)
    
    # Transpose the DataFrame to get proper column structure
    new_df = new_df.T
    
    return new_df

def process_avg_per_sample_norm_confusion_matrix_total(df):
    """
    Process data by normalizing confusion matrix values over the total of all CM_ columns.
    This gives the proportion of each CM_ value relative to the total word count across all categories.
    """
    eval_ids = df['eval_id'].unique()
    new_df = pd.DataFrame()
    for eval_id in eval_ids:
        eval_df = df[df['eval_id'] == eval_id]
        sample_data = eval_df.loc[:, 'eval_id':'treatment_strength'].iloc[0]
        category_data = eval_df.loc[:, 'answer_match':'which_treatment_mgf']
        CM_data = eval_df.loc[:, 'CM_in_quote_prefill':'CM_out_quote_inconsistent']
        
        # Calculate averages and standard deviations on raw data first
        category_data_row_avg = category_data.mean(axis=0)
        CM_data_row_avg = CM_data.mean(axis=0)
        category_data_row_std = category_data.std(axis=0)
        CM_data_row_std = CM_data.std(axis=0)
        
        # Now normalize the averaged CM data by the total of all CM_ columns
        CM_data_row_avg_normalized = CM_data_row_avg.copy()
        CM_data_row_std_normalized = CM_data_row_std.copy()
        
        # Calculate total from all CM_ columns
        total_cm = CM_data_row_avg.sum()
        
        # Normalize averages by total (avoid division by zero)
        if total_cm > 0:
            CM_data_row_avg_normalized = CM_data_row_avg / total_cm
            # Also normalize the standard deviations
            CM_data_row_std_normalized = CM_data_row_std / total_cm
        else:
            CM_data_row_avg_normalized = CM_data_row_avg * 0  # Set all to 0
            CM_data_row_std_normalized = CM_data_row_std * 0  # Set all to 0
        
        # Rename columns to append _avg and _std
        category_data_row_avg.index = [col + '_avg' for col in category_data_row_avg.index]
        CM_data_row_avg_normalized.index = [col + '_avg' for col in CM_data_row_avg_normalized.index]
        category_data_row_std.index = [col + '_std' for col in category_data_row_std.index]
        CM_data_row_std_normalized.index = [col + '_std' for col in CM_data_row_std_normalized.index]
        
        # Combine all data: original sample data + averaged category data + averaged CM data + std category data + std CM data
        new_row = pd.concat([sample_data, category_data_row_avg, category_data_row_std, CM_data_row_avg_normalized, CM_data_row_std_normalized], axis=0)
        new_df = pd.concat([new_df, new_row], axis=1)
    
    # Transpose the DataFrame to get proper column structure
    new_df = new_df.T
    
    return new_df

def filter_data(df, filter_dict):
    new_df = df.copy()
    for key, value in filter_dict.items():
        new_df = new_df[new_df[key] == value]
    return new_df

EXPERIMENT_NAME = "wikihow_summary_injection_conf_mat"
CONTROL_LOG_DIR = f"logs/{EXPERIMENT_NAME}/control"
TREATMENT_LOG_DIR = f"logs/{EXPERIMENT_NAME}/treatment"
control_evals_df = extract_treatment_data_with_conf_matrix(CONTROL_LOG_DIR)
treatment_evals_df = extract_treatment_data_with_conf_matrix(TREATMENT_LOG_DIR)
save_per_trial = False
save_per_sample = False
save_per_sample_norm_confusion_matrix_rows = False
save_per_sample_norm_confusion_matrix_columns = False
save_per_sample_norm_confusion_matrix_rows_filtered = False
save_per_sample_norm_confusion_matrix_columns_filtered = False
save_per_sample_norm_confusion_matrix_total_filtered = True

if save_per_trial == True:
    if control_evals_df is not None:
        print(f"Control samples shape: {control_evals_df.shape}")
        print(f"Control columns: {list(control_evals_df.columns)}")

    if treatment_evals_df is not None:
        print(f"Treatment samples shape: {treatment_evals_df.shape}")
        print(f"Treatment columns: {list(treatment_evals_df.columns)}")

    control_save_filename = f"scripts/data/{EXPERIMENT_NAME}/control/evals_df_per_trial.csv"
    treatment_save_filename = f"scripts/data/{EXPERIMENT_NAME}/treatment/evals_df_per_trial.csv"


    if control_evals_df is not None:
        os.makedirs(os.path.dirname(control_save_filename), exist_ok=True)
        print(f"Created directory: {os.path.dirname(control_save_filename)}")
        control_evals_df.to_csv(control_save_filename, index=False)
        print(f"Saved control data to: {control_save_filename}")

    if treatment_evals_df is not None:
        os.makedirs(os.path.dirname(treatment_save_filename), exist_ok=True)
        print(f"Created directory: {os.path.dirname(treatment_save_filename)}")
        treatment_evals_df.to_csv(treatment_save_filename, index=False)
        print(f"Saved treatment data to: {treatment_save_filename}")

if save_per_sample == True:
    """if control_evals_df is not None:
        avg_control_evals_df = process_avg_per_sample(control_evals_df)
        avg_control_evals_df.to_csv(f"scripts/data/{EXPERIMENT_NAME}/control/avg_evals_df_per_sample.csv", index=False)"""
        
    if treatment_evals_df is not None:
        avg_treatment_evals_df = process_avg_per_sample(treatment_evals_df)
        avg_treatment_evals_df.to_csv(f"scripts/data/{EXPERIMENT_NAME}/treatment/avg_evals_df_per_sample.csv", index=False)

if save_per_sample_norm_confusion_matrix_rows == True:
    if treatment_evals_df is not None:
        avg_treatment_evals_df = process_avg_per_sample_norm_confusion_matrix_rows(treatment_evals_df)
        avg_treatment_evals_df.to_csv(f"scripts/data/{EXPERIMENT_NAME}/treatment/avg_evals_df_per_sample_norm_confusion_matrix_rows.csv", index=False)

if save_per_sample_norm_confusion_matrix_columns == True:
    if treatment_evals_df is not None:
        avg_treatment_evals_df = process_avg_per_sample_norm_confusion_matrix_columns(treatment_evals_df)
        avg_treatment_evals_df.to_csv(f"scripts/data/{EXPERIMENT_NAME}/treatment/avg_evals_df_per_sample_norm_confusion_matrix_columns.csv", index=False)

if save_per_sample_norm_confusion_matrix_rows_filtered == True:
    if treatment_evals_df is not None:
        avg_treatment_evals_df = process_avg_per_sample_norm_confusion_matrix_rows(treatment_evals_df)

        caps_IL20_avg_treatment_evals_df = filter_data(avg_treatment_evals_df, {'injection_length': "20", 'treatment_type': 'capitalization'})
        caps_IL20_avg_treatment_evals_df.to_csv(f"scripts/data/{EXPERIMENT_NAME}/treatment/avg_evals_df_per_sample_norm_confusion_matrix_rows_caps_IL20.csv", index=False)

        typo_IL20_avg_treatment_evals_df = filter_data(avg_treatment_evals_df, {'injection_length': "20", 'treatment_type': 'typo'})
        typo_IL20_avg_treatment_evals_df.to_csv(f"scripts/data/{EXPERIMENT_NAME}/treatment/avg_evals_df_per_sample_norm_confusion_matrix_rows_typo_IL20.csv", index=False)

if save_per_sample_norm_confusion_matrix_columns_filtered == True:
    if treatment_evals_df is not None:
        avg_treatment_evals_df = process_avg_per_sample_norm_confusion_matrix_columns(treatment_evals_df)

        caps_IL20_avg_treatment_evals_df = filter_data(avg_treatment_evals_df, {'injection_length': "20", 'treatment_type': 'capitalization'})
        caps_IL20_avg_treatment_evals_df.to_csv(f"scripts/data/{EXPERIMENT_NAME}/treatment/avg_evals_df_per_sample_norm_confusion_matrix_columns_caps_IL20.csv", index=False)
        
        typo_IL20_avg_treatment_evals_df = filter_data(avg_treatment_evals_df, {'injection_length': "20", 'treatment_type': 'typo'})
        typo_IL20_avg_treatment_evals_df.to_csv(f"scripts/data/{EXPERIMENT_NAME}/treatment/avg_evals_df_per_sample_norm_confusion_matrix_columns_typo_IL20.csv", index=False)

if save_per_sample_norm_confusion_matrix_total_filtered == True:
    if treatment_evals_df is not None:
        avg_treatment_evals_df = process_avg_per_sample_norm_confusion_matrix_total(treatment_evals_df)

        caps_IL20_avg_treatment_evals_df = filter_data(avg_treatment_evals_df, {'injection_length': "20", 'treatment_type': 'capitalization'})
        caps_IL20_avg_treatment_evals_df.to_csv(f"scripts/data/{EXPERIMENT_NAME}/treatment/avg_evals_df_per_sample_norm_confusion_matrix_total_caps_IL20.csv", index=False)
        
        typo_IL20_avg_treatment_evals_df = filter_data(avg_treatment_evals_df, {'injection_length': "20", 'treatment_type': 'typo'})
        typo_IL20_avg_treatment_evals_df.to_csv(f"scripts/data/{EXPERIMENT_NAME}/treatment/avg_evals_df_per_sample_norm_confusion_matrix_total_typo_IL20.csv", index=False)

"""filtered_sample_df = filter_data(sample_df, {'injection_length': "20", 'treatment_type': 'capitalization'})
print(filtered_sample_df.head())"""

