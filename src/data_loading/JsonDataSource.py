from src.data_loading.table_extractor import get_create_table_blocks, get_sqlite_schemas
from src.data_loading.utils import load_json_files, SPIDER_PATH
import os
from tqdm import tqdm

class JsonDataSource:
    def __init__(self, json_paths, keep_fields=("db_id", "query", "question")):
        self.records = load_json_files(json_paths)
        self.keep_fields = keep_fields
        self.db_schemas = {}

        # Pre-load all schemas into memory to keep __getitem__ fast and stateless
        spider_path = os.environ.get(SPIDER_PATH)
        if spider_path is None:
            raise Exception("SPIDER_PATH environment variable not set")
        for raw in self.records:
            db_id = raw.get("db_id")
            if db_id and db_id not in self.db_schemas:
                file_path = f'{spider_path}database/{db_id}/{db_id}.sqlite'
                t_path = f'{spider_path}test_database/{db_id}/{db_id}.sqlite'
                if os.path.exists(file_path):
                    self.db_schemas[db_id] = "\n".join(get_sqlite_schemas(file_path))
                elif os.path.exists(t_path):
                    self.db_schemas[db_id] = "\n".join(get_sqlite_schemas(t_path))
                else:
                    raise FileNotFoundError(f"Database file not found: {file_path}")

    def _create_prompt(self, record):
        db_id = record["db_id"]
        schema = self.db_schemas.get(db_id, "")
        question = record["question"]

        # Format the prompt
        return (
            f"<start_of_turn>user\n"
            f"Based on this schema:\n{schema}\n\n"
            f"Write a SQL query to answer this question: {question}\n"
            f"using an sql.\n"
            f"<end_of_turn>\n<start_of_turn>model\n"
        )

    def filter_by_prompt_length(self, max_len):
        """Filters the records in-place based on the generated prompt length."""
        if not isinstance(max_len, int) or max_len <= 0:
            return

        self.records = [
            record for record in self.records if len(self._create_prompt(record)) <= max_len
        ]

    def filter_by_token_length(self, tokenizer, max_len):
        """Filters the records in-place based on the tokenized prompt length."""
        if not hasattr(tokenizer, 'encode_as_ids') or not callable(tokenizer.encode_as_ids):
            raise ValueError("Tokenizer must have an 'encode_as_ids' method.")
        if not isinstance(max_len, int) or max_len <= 0:
            return

        initial_count = len(self.records)
        
        # This can be slow, so we show progress
        self.records = [
            record for record in tqdm(self.records, desc="Filtering by token length") 
            if len(tokenizer.encode_as_ids(self._create_prompt(record))) <= max_len
        ]
        
        final_count = len(self.records)
        print(f"Filtered from {initial_count} to {final_count} records (kept {final_count / initial_count:.2%}).")


    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        raw = self.records[index]
        prompt = self._create_prompt(raw)

        return {
            "db_id": raw["db_id"],
            "prompts": prompt,
            "ground_truth": raw.get("query", "")
        }