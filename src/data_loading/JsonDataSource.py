from src.data_loading.table_extractor import get_create_table_blocks
from src.data_loading.utils import load_json_files, SPIDER_PATH
import os

class JsonDataSource:
    def __init__(self, json_paths, keep_fields=("db_id", "query", "question")):
        self.records = load_json_files(json_paths)
        self.keep_fields = keep_fields
        self.db_records = {}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        raw = self.records[index]

        record = {
            key: raw[key]
            for key in self.keep_fields
            if key in raw
        }

        file_path = f'{os.environ[SPIDER_PATH]}/database/{record["db_id"]}/schema.sql'
        if file_path not in self.db_records:
            record["db_definitions"] = "\n".join(get_create_table_blocks(file_path))
            self.db_records[file_path] = record["db_definitions"]
        else:
            record["db_definitions"] = self.db_records[file_path]
        return record
