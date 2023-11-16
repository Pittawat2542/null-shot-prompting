import json
import os
import random
import re
from abc import ABC
from pathlib import Path

from loguru import logger

from src.config import DATASETS_DIRECTORY, NUM_FEW_SHOT_SAMPLES
from src.models.data_item import DataItem
from src.tasks.task import Task


class StrategyQA(Task, ABC):
    """StrategyQA"""

    dataset_path = Path(os.getcwd()) / DATASETS_DIRECTORY / 'StrategyQA.json'
    dev_dataset_path = Path(os.getcwd()) / DATASETS_DIRECTORY / 'StrategyQA_train.json'

    def __new__(cls, *args, **kwargs):
        if cls is StrategyQA:
            raise TypeError(f"'{cls.__name__}' cannot be instantiated")
        return object.__new__(cls)

    @classmethod
    def has_native_cot_samples_supported(cls) -> bool:
        return True

    @classmethod
    def get_few_shot_samples(cls) -> list[DataItem]:
        dev_list = []
        with open(cls.dev_dataset_path) as dataset:
            data = json.load(dataset)
            for item in data:
                parsed_item = DataItem(f"Question: {item['question']}\nChoices: A) True, B) False\nAnswer:",
                                       " ".join(item["facts"]),
                                       "A" if item["answer"] is True else "B")
                dev_list.append(parsed_item)
        few_shot_examples = random.sample(dev_list, NUM_FEW_SHOT_SAMPLES)
        return few_shot_examples

    @classmethod
    def get_task(cls, item: dict) -> DataItem:
        choices = "A) True, B) False"
        return DataItem(f"Question: {item['input']}\nChoices: {choices}\nAnswer:", item["target"],
                        "A" if item["target_scores"]["Yes"] == 1 else "B")

    @classmethod
    def get_task_list(cls, data_path=dataset_path) -> list[DataItem]:
        task_list = []
        with open(data_path) as dataset:
            data = json.load(dataset)
            for item in data['examples']:
                parsed_item = cls.get_task(item)
                task_list.append(parsed_item)
        return task_list

    @classmethod
    def evaluate(cls, response: str, answer: str) -> (bool, str):
        pattern = r"([A-B]\))|(False|True)"
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
            prediction = extracted_answer.group(1) if extracted_answer.group(
                1) is not None else extracted_answer.group(0)
            if prediction == "True":
                prediction = "A"
            elif prediction == "False":
                prediction = "B"
            else:
                prediction = prediction[0]
            logger.debug(f"Prediction: {prediction}, Answer: {answer}")
            return prediction == answer, prediction
        logger.debug(f"Could not extract prediction from response")
        return False, ""

    def __str__(self) -> str:
        return "StrategyQA"
