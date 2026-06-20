import grain
import os
from src.data_loading.JsonDataSource import JsonDataSource
from src.data_loading.utils import SPIDER_PATH
from dotenv import load_dotenv

load_dotenv()

# Ensure the environment variable is loaded before it's used.
base_path = os.environ.get(SPIDER_PATH)
if not base_path:
    raise ValueError(f"{SPIDER_PATH} environment variable not set. Please check your .env file.")


def get_sources():
    """Initializes and returns the raw data sources without any filtering."""
    train_source = JsonDataSource([
        os.path.join(base_path, "train_others.json"),
        os.path.join(base_path, "train_spider.json"),
    ])

    dev_source = JsonDataSource([
        os.path.join(base_path, "dev.json"),
    ])

    test_source = JsonDataSource([
        os.path.join(base_path, "test.json"),
    ])
    
    return train_source, dev_source, test_source


def create_dataloaders(train_source, dev_source, test_source, tokenizer=None, max_token_len=None):
    """
    Creates Grain data loaders from data sources, with optional token-based filtering.
    
    Args:
        train_source: The training data source.
        dev_source: The development/validation data source.
        test_source: The test data source.
        tokenizer: A tokenizer instance (e.g., SentencePieceProcessor) with an 
                   'encode_as_ids' method. Required if max_token_len is set.
        max_token_len: The maximum number of tokens for a prompt to be included.
    """
    if max_token_len is not None:
        if tokenizer is None:
            raise ValueError("A tokenizer must be provided to filter by token length.")
        
        print(f"Filtering data sources to a max token length of {max_token_len}...")
        train_source.filter_by_token_length(tokenizer, max_token_len)
        dev_source.filter_by_token_length(tokenizer, max_token_len)
        test_source.filter_by_token_length(tokenizer, max_token_len)

    train_loader = grain.load(
        train_source,
        num_epochs=None,  # Loop indefinitely for training
        shuffle=True,
        seed=42,
        batch_size=32,
        worker_count=0,
    )

    dev_loader = grain.load(
        dev_source,
        num_epochs=1,  # Single pass for evaluation
        shuffle=False,
        batch_size=1,
        worker_count=0,
    )

    test_loader = grain.load(
        test_source,
        num_epochs=1,
        shuffle=False,
        batch_size=32,
        worker_count=0,
    )

    return train_loader, dev_loader, test_loader
