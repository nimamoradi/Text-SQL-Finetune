import json

SPIDER_PATH = "spider_path"

def load_json_files(paths):
    records = []
    for path in paths:
        with open(path, "r") as f:
            data = json.load(f)

        # If each file is a list of records:
        records.extend(data)

    return records

