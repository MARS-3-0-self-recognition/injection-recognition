import os
import json
import random
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging

# Import our text modification functions
from string_modifier import randomly_capitalize_string, introduce_typos

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WikiArticleProcessor:
    """
    A class to load and process articles from the WikiSum dataset.
    
    The WikiSum dataset typically contains Wikihow articles with their summaries.
    This processor can load articles and apply various text modifications.
    """
    
    def __init__(self, dataset_path: Optional[str] = None):
        """
        Initialize the WikiArticleProcessor.
        
        Args:
            dataset_path (Optional[str]): Path to the WikiSum dataset directory
        """
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.articles = []
        self.processed_articles = []
        
    def load_wikisum_dataset(self, dataset_path: Optional[str] = None, use_huggingface: bool = False) -> List[Dict]:
        """
        Load articles from the WikiSum dataset.
        
        Args:
            dataset_path (Optional[str]): Path to the dataset directory. If None, uses self.dataset_path
            use_huggingface (bool): If True, load from Hugging Face datasets library
            
        Returns:
            List[Dict]: List of loaded articles with their metadata
        """
        if use_huggingface:
            return self._load_from_huggingface()
        
        if dataset_path:
            self.dataset_path = Path(dataset_path)
        
        if not self.dataset_path or not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")
        
        logger.info(f"Loading WikiSum dataset from: {self.dataset_path}")
        
        articles = []
        
        # Look for common WikiSum dataset file patterns
        possible_files = [
            "train.json",
            "validation.json", 
            "test.json",
            "articles.json",
            "wikisum.json",
            "sample_articles.json"  # For our sample dataset
        ]
        
        loaded_files = []
        
        for filename in possible_files:
            file_path = self.dataset_path / filename
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            articles.extend(data)
                        elif isinstance(data, dict) and 'articles' in data:
                            articles.extend(data['articles'])
                        else:
                            articles.append(data)
                        loaded_files.append(filename)
                        logger.info(f"Loaded {len(data) if isinstance(data, list) else 1} articles from {filename}")
                except Exception as e:
                    logger.warning(f"Failed to load {filename}: {e}")
        
        # If no JSON files found, try to load from text files
        if not loaded_files:
            articles = self._load_from_text_files()
        
        self.articles = articles
        logger.info(f"Total articles loaded: {len(self.articles)}")
        return articles
    
    def _load_from_huggingface(self) -> List[Dict]:
        """
        Load articles from Hugging Face WikiSum dataset.
        
        Returns:
            List[Dict]: List of loaded articles with their metadata
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install the datasets library: pip install datasets")
        
        logger.info("Loading WikiSum dataset from Hugging Face...")
        
        # Load the dataset
        dataset = load_dataset("d0rj/wikisum")
        
        articles = []
        
        # Convert to our format
        for split in ['train', 'validation', 'test']:
            if split in dataset:
                split_data = dataset[split]
                try:
                    split_len = len(split_data)
                except TypeError:
                    split_len = "unknown"
                logger.info(f"Processing {split} split with {split_len} articles")
                
                for i, item in enumerate(split_data):
                    article = {
                        'id': f"{split}_{i}",
                        'title': item.get('title', ''),
                        'text': item.get('article', ''),  # Full article text
                        'summary': item.get('summary', ''),  # Summary
                        'url': item.get('url', ''),
                        'step_headers': item.get('step_headers', ''),
                        'split': split
                    }
                    articles.append(article)
        
        self.articles = articles
        logger.info(f"Total articles loaded from Hugging Face: {len(self.articles)}")
        return articles
    
    def _load_from_text_files(self) -> List[Dict]:
        """
        Load articles from text files if JSON files are not found.
        
        Returns:
            List[Dict]: List of articles loaded from text files
        """
        articles = []
        
        # Look for .txt files
        if self.dataset_path is not None:
            txt_files = list(self.dataset_path.glob("*.txt"))
            
            for txt_file in txt_files:
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            articles.append({
                                'id': txt_file.stem,
                                'title': txt_file.stem.replace('_', ' ').title(),
                                'text': content,
                                'summary': content[:200] + "..." if len(content) > 200 else content,
                                'source_file': str(txt_file)
                            })
                except Exception as e:
                    logger.warning(f"Failed to load {txt_file}: {e}")
        
        return articles
    
    def get_article_by_id(self, article_id: str) -> Optional[Dict]:
        """
        Get a specific article by its ID.
        
        Args:
            article_id (str): The ID of the article to retrieve
            
        Returns:
            Optional[Dict]: The article if found, None otherwise
        """
        for article in self.articles:
            if article.get('id') == article_id:
                return article
        return None
    
    def get_random_articles(self, count: int = 5) -> List[Dict]:
        """
        Get a random sample of articles.
        
        Args:
            count (int): Number of articles to retrieve
            
        Returns:
            List[Dict]: Random sample of articles
        """
        if not self.articles:
            raise ValueError("No articles loaded. Call load_wikisum_dataset() first.")
        
        count = min(count, len(self.articles))
        return random.sample(self.articles, count)
    
    def apply_text_modifications(self, 
                               articles: Optional[List[Dict]] = None,
                               capitalize_percentage: int = 30,
                               typo_rates: Optional[Dict[str, int]] = None) -> List[Dict]:
        """
        Apply text modifications to articles.
        
        Args:
            articles (Optional[List[Dict]]): Articles to process. If None, uses all loaded articles
            capitalize_percentage (int): Percentage of letters to randomly capitalize
            typo_rates (Optional[Dict[str, int]]): Dictionary with typo rates:
                - flip_rate: Percentage of adjacent letter pairs to flip
                - drop_rate: Percentage of characters to drop
                - add_rate: Percentage of positions to add random characters
                - substitute_rate: Percentage of characters to substitute
        
        Returns:
            List[Dict]: Articles with modified text
        """
        if articles is None:
            articles = self.articles
        
        if not articles:
            raise ValueError("No articles provided or loaded.")
        
        # Default typo rates if not provided
        if typo_rates is None:
            typo_rates = {
                'flip_rate': 8,
                'drop_rate': 5,
                'add_rate': 3,
                'substitute_rate': 12
            }
        
        processed_articles = []
        
        for article in articles:
            processed_article = article.copy()
            
            # Get the text to modify (prefer 'text' field, fallback to 'summary')
            original_text = article.get('text', article.get('summary', ''))
            
            if not original_text:
                logger.warning(f"No text found for article {article.get('id', 'unknown')}")
                processed_articles.append(processed_article)
                continue
            
            # Apply capitalization
            if capitalize_percentage > 0:
                capitalized_text = randomly_capitalize_string(original_text, capitalize_percentage)
                processed_article['capitalized_text'] = capitalized_text
            
            # Apply typos
            if any(rate > 0 for rate in typo_rates.values()):
                typo_text = introduce_typos(
                    original_text,
                    flip_rate=typo_rates.get('flip_rate', 0),
                    drop_rate=typo_rates.get('drop_rate', 0),
                    add_rate=typo_rates.get('add_rate', 0),
                    substitute_rate=typo_rates.get('substitute_rate', 0)
                )
                processed_article['typo_text'] = typo_text
            
            # Apply both modifications
            if capitalize_percentage > 0 and any(rate > 0 for rate in typo_rates.values()):
                # Apply typos first, then capitalization
                combined_text = introduce_typos(
                    original_text,
                    flip_rate=typo_rates.get('flip_rate', 0),
                    drop_rate=typo_rates.get('drop_rate', 0),
                    add_rate=typo_rates.get('add_rate', 0),
                    substitute_rate=typo_rates.get('substitute_rate', 0)
                )
                combined_text = randomly_capitalize_string(combined_text, capitalize_percentage)
                processed_article['combined_modified_text'] = combined_text
            
            processed_articles.append(processed_article)
        
        self.processed_articles = processed_articles
        return processed_articles
    
    def save_processed_articles(self, output_path: str, format: str = 'json') -> None:
        """
        Save processed articles to a file.
        
        Args:
            output_path (str): Path to save the processed articles
            format (str): Output format ('json' or 'txt')
        """
        if not self.processed_articles:
            raise ValueError("No processed articles to save. Call apply_text_modifications() first.")
        
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'json':
            with open(output_path_obj, 'w', encoding='utf-8') as f:
                json.dump(self.processed_articles, f, indent=2, ensure_ascii=False)
        elif format.lower() == 'txt':
            with open(output_path_obj, 'w', encoding='utf-8') as f:
                for article in self.processed_articles:
                    f.write(f"Article ID: {article.get('id', 'unknown')}\n")
                    f.write(f"Title: {article.get('title', 'unknown')}\n")
                    f.write(f"Original Text: {article.get('text', article.get('summary', ''))}\n")
                    
                    if 'capitalized_text' in article:
                        f.write(f"Capitalized Text: {article['capitalized_text']}\n")
                    
                    if 'typo_text' in article:
                        f.write(f"Typo Text: {article['typo_text']}\n")
                    
                    if 'combined_modified_text' in article:
                        f.write(f"Combined Modified Text: {article['combined_modified_text']}\n")
                    
                    f.write("-" * 80 + "\n\n")
        else:
            raise ValueError("Unsupported format. Use 'json' or 'txt'.")
        
        logger.info(f"Processed articles saved to: {output_path_obj}")
    
    def print_article_comparison(self, article_index: int = 0) -> None:
        """
        Print a comparison of original and modified text for a specific article.
        
        Args:
            article_index (int): Index of the article to display
        """
        if not self.processed_articles:
            raise ValueError("No processed articles. Call apply_text_modifications() first.")
        
        if article_index >= len(self.processed_articles):
            raise ValueError(f"Article index {article_index} out of range.")
        
        article = self.processed_articles[article_index]
        
        print(f"\n{'='*80}")
        print(f"ARTICLE COMPARISON - {article.get('title', 'Unknown Title')}")
        print(f"{'='*80}")
        print(f"Article ID: {article.get('id', 'unknown')}")
        print(f"Source: {article.get('source_file', 'unknown')}")
        print()
        
        original_text = article.get('text', article.get('summary', ''))
        print(f"ORIGINAL TEXT:")
        print(f"{'─'*40}")
        print(original_text[:500] + "..." if len(original_text) > 500 else original_text)
        print()
        
        if 'capitalized_text' in article:
            print(f"CAPITALIZED TEXT:")
            print(f"{'─'*40}")
            capitalized = article['capitalized_text']
            print(capitalized[:500] + "..." if len(capitalized) > 500 else capitalized)
            print()
        
        if 'typo_text' in article:
            print(f"TEXT WITH TYPOS:")
            print(f"{'─'*40}")
            typo = article['typo_text']
            print(typo[:500] + "..." if len(typo) > 500 else typo)
            print()
        
        if 'combined_modified_text' in article:
            print(f"COMBINED MODIFIED TEXT:")
            print(f"{'─'*40}")
            combined = article['combined_modified_text']
            print(combined[:500] + "..." if len(combined) > 500 else combined)
            print()


def create_sample_dataset(dataset_path: str = "sample_wikisum_data") -> None:
    """
    Create a sample WikiSum-like dataset for testing purposes.
    
    Args:
        dataset_path (str): Path where to create the sample dataset
    """
    sample_articles = [
        {
            "id": "sample_001",
            "title": "Artificial Intelligence",
            "text": "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of intelligent agents: any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.",
            "summary": "AI is machine intelligence that perceives environments and takes goal-directed actions."
        },
        {
            "id": "sample_002", 
            "title": "Machine Learning",
            "text": "Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves.",
            "summary": "Machine learning enables systems to learn from data without explicit programming."
        },
        {
            "id": "sample_003",
            "title": "Natural Language Processing",
            "text": "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data.",
            "summary": "NLP focuses on computer-human language interaction and processing."
        },
        {
            "id": "sample_004",
            "title": "Deep Learning",
            "text": "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep learning architectures such as deep neural networks, deep belief networks, recurrent neural networks and convolutional neural networks have been applied to fields including computer vision, speech recognition, natural language processing, audio recognition, social network filtering, machine translation, bioinformatics, drug design, medical image analysis, material inspection and board game programs, where they have produced results comparable to and in some cases surpassing human expert performance.",
            "summary": "Deep learning uses neural networks for representation learning across various domains."
        },
        {
            "id": "sample_005",
            "title": "Computer Vision",
            "text": "Computer vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images or videos. From the perspective of engineering, it seeks to understand and automate tasks that the human visual system can do. Computer vision tasks include methods for acquiring, processing, analyzing and understanding digital images, and extraction of high-dimensional data from the real world in order to produce numerical or symbolic information.",
            "summary": "Computer vision enables computers to understand and process visual information from images and videos."
        }
    ]
    
    dataset_path_obj = Path(dataset_path)
    dataset_path_obj.mkdir(exist_ok=True)
    
    with open(dataset_path_obj / "sample_articles.json", 'w', encoding='utf-8') as f:
        json.dump(sample_articles, f, indent=2, ensure_ascii=False)
    
    print(f"Sample dataset created at: {dataset_path}")
    print(f"Created {len(sample_articles)} sample articles")


# Example usage and testing
if __name__ == "__main__":
    # Create a sample dataset for testing
    print("Creating sample dataset...")
    create_sample_dataset()
    
    # Initialize the processor
    processor = WikiArticleProcessor()
    
    # Load the sample dataset
    print("\nLoading sample dataset...")
    articles = processor.load_wikisum_dataset("sample_wikisum_data")
    
    # Apply text modifications
    print("\nApplying text modifications...")
    processed_articles = processor.apply_text_modifications(
        articles=articles,
        capitalize_percentage=40,
        typo_rates={
            'flip_rate': 10,
            'drop_rate': 5,
            'add_rate': 3,
            'substitute_rate': 15
        }
    )
    
    # Print comparison for the first article
    processor.print_article_comparison(0)
    
    # Save processed articles
    print("\nSaving processed articles...")
    processor.save_processed_articles("processed_articles.json", format='json')
    processor.save_processed_articles("processed_articles.txt", format='txt')
    
    print("\nProcessing complete!") 