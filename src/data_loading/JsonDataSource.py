from src.data_loading.table_extractor import get_create_table_blocks, get_sqlite_schemas
from src.data_loading.utils import load_json_files, SPIDER_PATH
import os

class JsonDataSource:
    def __init__(self, json_paths, keep_fields=("db_id", "query", "question")):
        self.records = load_json_files(json_paths)
        self.keep_fields = keep_fields
        self.db_schemas = {}

        # Pre-load all schemas into memory to keep __getitem__ fast and stateless
        spider_path = os.environ[SPIDER_PATH]
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
            f"You are a SQL expert. Based on this schema:\n{schema}\n\n"
            f"Write a SQL query to answer this question: {question}\n"
            f"<end_of_turn>\n<start_of_turn>model\n"
        )

    def filter_by_prompt_length(self, max_len):
        """Filters the records in-place based on the generated prompt length."""
        if not isinstance(max_len, int) or max_len <= 0:
            return

        self.records = [
            record for record in self.records if len(self._create_prompt(record)) <= max_len
        ]

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