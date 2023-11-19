import os
import random
from abc import ABC
from pathlib import Path

from jsonlines import jsonlines
from loguru import logger

from src.config import DATASETS_DIRECTORY, NUM_FEW_SHOT_SAMPLES
from src.models.data_item import DataItem
from src.tasks.task import Task


class WMTENJA(Task, ABC):
    """WMTENJA"""

    dataset_path = Path(os.getcwd()) / DATASETS_DIRECTORY / 'WMTJAEN.jsonl'
    dev_dataset_path = Path(os.getcwd()) / DATASETS_DIRECTORY / 'WMTJAEN_dev.jsonl'

    def __new__(cls, *args, **kwargs):
        if cls is WMTENJA:
            raise TypeError(f"'{cls.__name__}' cannot be instantiated")
        return object.__new__(cls)

    @classmethod
    def has_native_cot_samples_supported(cls) -> bool:
        return False

    @classmethod
    def get_few_shot_samples(cls) -> list[DataItem]:
        dev_list = cls.get_task_list(cls.dev_dataset_path)
        few_shot_examples = random.sample(dev_list, NUM_FEW_SHOT_SAMPLES)
        for sample in few_shot_examples:
            sample.rationale = ""
        return few_shot_examples

    @classmethod
    def get_task(cls, item: dict) -> DataItem:
        return DataItem(f"Original: {item['en']}\nJapanese Translation:", None,
                        item["ja"])

    @classmethod
    def get_task_list(cls, data_path=dataset_path) -> list[DataItem]:
        task_list = []
        with jsonlines.open(data_path) as dataset:
            for item in dataset:
                parsed_item = cls.get_task(item)
                task_list.append(parsed_item)
        return task_list

    @classmethod
    def evaluate(cls, response: str, answer: str) -> (bool, str):
        if len(response) == 0:
            logger.debug(f"Could not extract prediction from response as response is empty")
            return False, ""

        lines = response.splitlines()
        first_line = lines[0]
        last_line = lines[-1]

        prediction = first_line if first_line == answer else last_line
        logger.debug(f"Prediction: {prediction}, Answer: {answer}")
        return prediction == answer, prediction

    def __str__(self) -> str:
        return "WMTENJA"
