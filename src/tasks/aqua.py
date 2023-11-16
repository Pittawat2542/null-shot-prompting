import os
import random
import re
from abc import ABC
from pathlib import Path

from jsonlines import jsonlines
from loguru import logger

from src.config import DATASETS_DIRECTORY, NUM_FEW_SHOT_SAMPLES
from src.models.data_item import DataItem
from src.tasks.task import Task


class AQuA(Task, ABC):
    """AQuA"""

    dataset_path = Path(os.getcwd()) / DATASETS_DIRECTORY / 'AQuA.jsonl'
    dev_dataset_path = Path(os.getcwd()) / DATASETS_DIRECTORY / 'AQuA_dev.jsonl'

    def __new__(cls, *args, **kwargs):
        if cls is AQuA:
            raise TypeError(f"'{cls.__name__}' cannot be instantiated")
        return object.__new__(cls)

    @classmethod
    def has_native_cot_samples_supported(cls) -> bool:
        return True

    @classmethod
    def get_few_shot_samples(cls) -> list[DataItem]:
        dev_list = cls.get_task_list(cls.dev_dataset_path)
        few_shot_examples = random.sample(dev_list, NUM_FEW_SHOT_SAMPLES)
        return few_shot_examples

    @classmethod
    def get_task(cls, item: dict) -> DataItem:
        choices = ", ".join(item["options"])
        return DataItem(f"Question: {item['question']}\nChoices: {choices}\nAnswer:", item["rationale"],
                        item["correct"])

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
        pattern = r"([A-E]\))"
        if len(response) == 0:
            logger.debug(f"Could not extract prediction from response as response is empty")
            return False, ""

        if len(response) == 1 and response.isupper():
            logger.debug(f"Prediction: {response}, Answer: {answer}")
            return response == answer, response

        lines = response.splitlines()
        first_line = lines[0]
        last_line = lines[-1]
        if re.search(pattern, last_line) is not None:
            extracted_answer = re.search(pattern, last_line)
        elif re.search(pattern, first_line) is not None:
            extracted_answer = re.search(pattern, first_line)
        else:
            extracted_answer = None

        if extracted_answer is not None:
            prediction = extracted_answer.group(1)[0]
            logger.debug(f"Prediction: {prediction}, Answer: {answer}")
            return prediction == answer, prediction
        logger.debug(f"Could not extract prediction from response")
        return False, ""

    def __str__(self) -> str:
        return "AQuA"
