import grain

from src.data_loading.JsonDataSource import JsonDataSource
import os

from src.data_loading.utils import SPIDER_PATH
from dotenv import load_dotenv

load_dotenv()

base_path = os.environ[SPIDER_PATH]


def get_sources():
    train_source = JsonDataSource([
        base_path + "train_others.json",
        base_path + "train_spider.json",
    ])

    dev_source = JsonDataSource([
        base_path + "dev.json",
    ])

    test_source = JsonDataSource([
        base_path + "test.json",
    ])
    return train_source, dev_source, test_source


def create_dataloaders(train_source, dev_source, test_source):
    train_loader = grain.load(
        train_source,
        num_epochs=None,
        shuffle=True,
        seed=42,
        batch_size=32,
        worker_count=0,
    )

    dev_loader = grain.load(
        dev_source,
        num_epochs=1,
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