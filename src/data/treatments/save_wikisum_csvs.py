#!/usr/bin/env python3
"""
Simple script to save random WikiSum articles to CSV files.

Usage:
    python save_wikisum_csvs.py <n> <save_path>

Examples:
    python save_wikisum_csvs.py 10 random_articles.csv
    python save_wikisum_csvs.py 5 /path/to/my_articles.csv
"""

import sys
import argparse
from pathlib import Path

# Add the current directory to the Python path to import our modules
import sys
sys.path.append(str(Path(__file__).parent))

from wikisum_utils import get_WikiSum_random


def main():
    """Main function to handle command line arguments and save WikiSum articles."""
    
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Save random WikiSum articles to a CSV file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s 10 random_articles.csv
  %(prog)s 5 /path/to/my_articles.csv
  %(prog)s 20 articles.csv --seed 42
        """
    )
    
    parser.add_argument(
        'n', 
        type=int, 
        help='Number of random articles to grab'
    )
    
    parser.add_argument(
        'save_path', 
        type=str, 
        help='Path where to save the CSV file'
    )
    
    parser.add_argument(
        '--seed', 
        type=int, 
        default=None,
        help='Random seed for reproducible results (optional)'
    )
    
    parser.add_argument(
        '--max-articles', 
        type=int, 
        default=None,
        help='Maximum number of articles to consider for random selection (optional)'
    )
    
    parser.add_argument(
        '--no-huggingface', 
        action='store_true',
        help='Do not use Hugging Face datasets library (use local files instead)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate arguments
    if args.n <= 0:
        print("Error: n must be a positive integer")
        sys.exit(1)
    
    # Convert save_path to Path object
    save_path = Path(args.save_path)
    
    # Ensure the directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Grabbing {args.n} random WikiSum articles...")
    print(f"Saving to: {save_path}")
    
    if args.seed:
        print(f"Using random seed: {args.seed}")
    
    try:
        # Get random articles
        df = get_WikiSum_random(
            n=args.n,
            max_articles=args.max_articles,
            use_huggingface=not args.no_huggingface,
            seed=args.seed,
            save_path=save_path
        )
        
        print(f"Successfully saved {len(df)} articles to {save_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 