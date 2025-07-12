#!/usr/bin/env python3
"""
Simplified data processing script for WikiSum dataset.

This script provides an easy way to:
1. Load WikiSum data by index range
2. Apply treatments with customizable parameters
3. Save results to CSV files

Edit the parameters below to customize your data processing.
"""

from wikisum_utils import get_WikiSum, apply_treatments_separate


def main():
    """Main processing function - edit parameters here."""
    
    print("=" * 60)
    print("WIKISUM DATA PROCESSING")
    print("=" * 60)
    
    # ========================================
    # EDIT THESE PARAMETERS AS NEEDED
    # ========================================
    
    # Data range parameters
    START_INDEX = 5    # Starting article index (inclusive)
    END_INDEX = 10     # Ending article index (exclusive)
    
    # Summary length parameters (percentages)
    SUMMARY_LENGTHS = [33, 66, 100]  # IL33, IL66, IL100
    
    # Treatment parameters
    TREATMENT_PARAMS = {
        'capitalization_rates': [25, 50, 75, 100],  # S1, S2, S3, S4
        
        # Option 1: Simple typo rates (applies to substitute_rate only)
        # 'typo_rates': [5, 10, 15],  # Simple list format
        
        # Option 2: Granular typo parameters (uncomment to use)
        # 'typo_rates': {
        #     'substitute_rate': [5, 10, 15],      # Character substitution rates
        #     'flip_rate': [3, 8, 12],             # Adjacent letter flip rates
        #     'drop_rate': [2, 5, 8],              # Character drop rates
        #     'add_rate': [1, 3, 5],               # Character addition rates
        # },
        
        # Option 3: Combined typo profiles (uncomment to use)
        # 'typo_rates': {
        #     'light':  {'substitute_rate': 5,  'flip_rate': 3,  'drop_rate': 2, 'add_rate': 1},
        #     'medium': {'substitute_rate': 10, 'flip_rate': 8,  'drop_rate': 5, 'add_rate': 3},
        #     'heavy':  {'substitute_rate': 15, 'flip_rate': 12, 'drop_rate': 8, 'add_rate': 5},
        # },
        
        # Option 4: Per-word typo rates (testing new functionality)
        'typo_rates': {
            'low':    {'typos_per_word': 0.2, 'typo_types': {'add_rate','substitute_rate', 'flip_rate', 'drop_rate'}},
            'medium': {'typos_per_word': 1.0, 'typo_types': {'substitute_rate', 'flip_rate', 'drop_rate'}},
            'high':   {'typos_per_word': 2.0, 'typo_types': {'substitute_rate', 'flip_rate', 'drop_rate', 'add_rate'}},
        },
        
        # 'other_treatments': {}  # Optional: add custom treatments
    }
    
    # ========================================
    # PROCESSING STARTS HERE
    # ========================================
    
    print(f"Processing articles {START_INDEX} to {END_INDEX-1}")
    print(f"Summary lengths: {SUMMARY_LENGTHS}")
    print(f"Treatment parameters: {TREATMENT_PARAMS}")
    print()
    
    # Step 1: Get WikiSum data
    print("Step 1: Loading WikiSum data...")
    try:
        df = get_WikiSum(START_INDEX, END_INDEX)
        print(f"✓ Loaded {len(df)} articles")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return
    
    # Step 2: Apply treatments and save to separate CSV files
    print("\nStep 2: Applying treatments to separate CSV files...")
    try:
        output_files = apply_treatments_separate(
            df=df,
            summary_lengths=SUMMARY_LENGTHS,
            treatment_params=TREATMENT_PARAMS,
            start_idx=START_INDEX,
            end_idx=END_INDEX
        )
        print("✓ All treatments applied and saved successfully")
    except Exception as e:
        print(f"✗ Error applying treatments: {e}")
        return
    
    # Summary
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"Input articles: {len(df)}")
    print(f"Output files created:")
    for treatment_name, file_path in output_files.items():
        print(f"  - {treatment_name}: {file_path}")
    print("\nYou can now:")
    print("  - Open the CSV files in Excel or other spreadsheet software")
    print("  - Modify the parameters above and run again")
    print("  - Add new treatment types by editing TREATMENT_PARAMS")


if __name__ == "__main__":
    main()

    print("\n=== TESTING get_WikiSum EXTENSIONS ===")
    from wikisum_utils import get_WikiSum, get_WikiSum_random

    # Test 1: Specific indices
    try:
        print("\nTest 1: get_WikiSum with indices=[0, 2, 4]")
        df_indices = get_WikiSum(indices=[0, 2, 4], use_huggingface=True)
        print(df_indices.head())
        print(f"Returned {len(df_indices)} articles.")
    except Exception as e:
        print(f"Test 1 failed: {e}")

    # Test 2: Random articles
    try:
        print("\nTest 2: get_WikiSum_random with n=3")
        df_random = get_WikiSum_random(n=3, use_huggingface=True, seed=42)
        print(df_random.head())
        print(f"Returned {len(df_random)} random articles.")
    except Exception as e:
        print(f"Test 2 failed: {e}")

    # Test 3: Custom save_path
    try:
        print("\nTest 3: get_WikiSum with indices=[1, 3] and custom save_path='custom_test_articles.csv'")
        df_custom = get_WikiSum(indices=[1, 3], use_huggingface=True, save_path="custom_test_articles.csv")
        print(df_custom.head())
        print(f"Returned {len(df_custom)} articles. Saved to custom_test_articles.csv")
    except Exception as e:
        print(f"Test 3 failed: {e}") 