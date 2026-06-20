import os
import sys
import argparse
import json
from tqdm import tqdm


# --- Direct Imports ---
import sentencepiece as spm
from src.data_loading.table_extractor import get_sqlite_schemas
from dotenv import load_dotenv

# --- Helper Functions ---
def load_json_files(json_paths):
    records = []
    for path in json_paths:
        with open(path, 'r') as f:
            records.extend(json.load(f))
    return records

def create_prompt(record, db_schemas):
    db_id = record["db_id"]
    schema = db_schemas.get(db_id, "")
    question = record["question"]
    return (
        f"<start_of_turn>user\n"
        f"You are a SQL expert. Based on this schema:\n{schema}\n\n"
        f"Write a SQL query to answer this question: {question}\n"
        f"<end_of_turn>\n<start_of_turn>model\n"
    )

# --- Main Script ---
def main(tokenizer_path: str, spider_path: str) -> None:
    """
    Loads data, calculates prompt token lengths, and saves them to a text file.
    This script does NOT perform any plotting.
    """
    print(f"Loading SentencePiece tokenizer from: {tokenizer_path}")
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(tokenizer_path)
    print("Tokenizer loaded successfully.")

    print("Loading data sources directly...")
    json_paths = [
        os.path.join(spider_path, "train_others.json"),
        os.path.join(spider_path, "train_spider.json"),
        os.path.join(spider_path, "dev.json"),
    ]
    records = load_json_files(json_paths)

    print("Pre-loading all database schemas...")
    db_schemas = {}
    for raw in tqdm(records, desc="Loading Schemas"):
        db_id = raw.get("db_id")
        if db_id and db_id not in db_schemas:
            db_file_path = os.path.join(spider_path, 'database', db_id, f'{db_id}.sqlite')
            if os.path.exists(db_file_path):
                db_schemas[db_id] = "\n".join(get_sqlite_schemas(db_file_path))
            else:
                test_db_path = os.path.join(spider_path, 'test_database', db_id, f'{db_id}.sqlite')
                if os.path.exists(test_db_path):
                    db_schemas[db_id] = "\n".join(get_sqlite_schemas(test_db_path))
                else:
                    print(f"Warning: Database file not found for db_id: {db_id}")

    print("Calculating prompt token lengths...")
    prompt_lengths = []
    for record in tqdm(records, desc="Tokenizing Prompts"):
        try:
            prompt_text = create_prompt(record, db_schemas)
            tokens = tokenizer.encode_as_ids(prompt_text)
            prompt_lengths.append(len(tokens))
        except Exception as e:
            print(f"Skipping an item due to error: {e}")

    # --- Save lengths to a file ---
    output_path = 'prompt_token_lengths.txt'
    with open(output_path, 'w') as f:
        for length in prompt_lengths:
            f.write(f"{length}\n")
    
    print(f"\nToken lengths saved to {output_path}")
    print("Now, run the plot_histogram.py script to generate the image.")

if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description='Analyze prompt token length distribution.')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to the SentencePiece tokenizer model file (.model).')
    default_spider_path = os.environ.get("SPIDER_PATH")
    parser.add_argument('--spider_path', type=str, default=default_spider_path, help='Path to the SPIDER dataset directory.')
    args = parser.parse_args()
    if not args.spider_path:
        raise ValueError("SPIDER_PATH must be set via .env or --spider_path.")
    main(args.tokenizer_path, args.spider_path)
