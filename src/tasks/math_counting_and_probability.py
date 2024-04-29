import os
import random
import re
from abc import ABC
from pathlib import Path

from jsonlines import jsonlines
from loguru import logger
from typing_extensions import List

from src.config import DATASETS_DIRECTORY, NUM_FEW_SHOT_SAMPLES
from src.models.data_item import DataItem
from src.tasks.task import Task


class MATHCountingAndProbability(Task, ABC):
    """MATH Counting and Probability"""

    dataset_path = Path(os.getcwd()) / DATASETS_DIRECTORY / 'MATH_counting_and_probability.jsonl'
    dev_dataset_path = Path(os.getcwd()) / DATASETS_DIRECTORY / 'MATH_counting_and_probability_train.jsonl'

    def __new__(cls, *args, **kwargs):
        if cls is MATHCountingAndProbability:
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
        return DataItem(f"Question: {item['problem']}\nAnswer: Output the final answer in \\boxed{{}} (LaTeX).",
                        item["solution"],
                        item["answers"])

    @classmethod
    def get_task_list(cls, data_path=dataset_path) -> list[DataItem]:
        task_list = []
        with jsonlines.open(data_path) as dataset:
            for item in dataset:
                parsed_item = cls.get_task(item)
                task_list.append(parsed_item)
        return task_list

    @classmethod
    def evaluate(cls, response: str, answer: str | List[str]) -> (bool, str):
        pattern = r"\\boxed\{(.+)\}"
        secondary_pattern = r"answer is \\boxed\{(.+)\}"

        if len(response) == 0:
            logger.debug(f"Could not extract prediction from response as response is empty")
            return False, ""

        lines = response.splitlines()
        first_line = lines[0]
        last_line = lines[-1]
        if len(re.findall(pattern, last_line)) > 0:
            prediction = re.findall(pattern, last_line)
        elif len(re.findall(secondary_pattern, last_line)) > 0:
            prediction = re.findall(secondary_pattern, last_line)
        elif len(re.findall(pattern, first_line)) > 0:
            prediction = re.findall(pattern, first_line)
        elif len(re.findall(secondary_pattern, first_line)) > 0:
            prediction = re.findall(secondary_pattern, first_line)
        else:
            prediction = None

        if prediction is not None and len(prediction) > 0:
            logger.debug(f"Prediction: {prediction}, Answer: {answer}")
            return any(ans in pred for ans in answer for pred in prediction), ", ".join(prediction)
        logger.debug(f"Could not extract prediction from response")
        return False, ""

    def __str__(self) -> str:
        return "MATH Counting and Probability"
