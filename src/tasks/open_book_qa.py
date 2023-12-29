import os
import re
from abc import ABC
from pathlib import Path

from jsonlines import jsonlines
from loguru import logger

from src.config import DATASETS_DIRECTORY
from src.models.data_item import DataItem
from src.tasks.task import Task


class OpenBookQA(Task, ABC):
    """OpenBookQA"""

    dataset_path = Path(os.getcwd()) / DATASETS_DIRECTORY / 'OpenBookQA.jsonl'
    dev_dataset_path = Path(os.getcwd()) / DATASETS_DIRECTORY / 'OpenBookQA_dev.jsonl'

    def __new__(cls, *args, **kwargs):
        if cls is OpenBookQA:
            raise TypeError(f"'{cls.__name__}' cannot be instantiated")
        return object.__new__(cls)

    @classmethod
    def has_native_cot_samples_supported(cls) -> bool:
        return False

    @classmethod
    def get_few_shot_samples(cls) -> list[DataItem]:
        # Manual
        few_shot_examples = [
            DataItem(
                "Question: From which distance will an object look the biggest?\nChoices: A) 4 feet, B) 30 feet, C) 1 kilometer, D) 1 mile\nAnswer:",
                "An object appears largest when it is closest to the observer. Out of all the choices, 4 feet is the closest. So, the answer is A.",
                "A"),
            DataItem(
                "Question: What could be used as an electrical conductor\nChoices: A) A penny, B) Shoe laces, C) Wood, D) A button\nAnswer:",
                "To conduct electricity, the object should be made of conductive metal. Out of all the choices, only a penny is typically made of metal. So, the answer is A.",
                "A"),
            DataItem(
                "Question: An animal can hunt by cracking open a\nChoices: A) claw, B) house, C) shell, D) bone\nAnswer:",
                "Cracking something open requires that a particular thing must be enclosed or closed. Out of all the choices, only a shell can enclose an animal. So, the answer is C.",
                "C"),
            DataItem(
                "Question: Tree rings can\nChoices: A) indicate how often the tree needs pruning, B) tell you what year a tree was planted, C) tell you how tall a tree is, D) indicate the year the tree will die\nAnswer:",
                "Tree rings indicate the age of trees, as one ring usually represents one year. Out of all the choices, tree rings can indicate the year a tree is planted. So, the answer is B.",
                "B"),
            DataItem(
                "Question: So many environments receive large amounts of rain, though an exception is\nChoices: A) the Sahara, B) the great plains, C) the rain-forest, D) Seattle\nAnswer:",
                "An environment that usually receives a smaller amount of rain is typically classified as a desert environment. Out of all the choices, only the Sahara is a desert. So, the answer is A.",
                "A"),
            DataItem(
                "Question: A river can disappear over a period of time?\nChoices: A) this is uncertain, B) this is sure, C) all of these, D) maybe\nAnswer:",
                "With constant use of water or changes in the territory around the river, it is possible that the river can disappear over a period of time. So, the answer is B.",
                "B"),
            DataItem(
                "Question: What can keep you moisturized other than water?\nChoices: A) Crackers, B) Sour Cream, C) cucumbers, D) Cheese\nAnswer:",
                "Foods containing a high amount of moisture can also help keep us hydrated. Out of all the choices, cucumbers contain the highest amount of liquid. So, the answer is C.",
                "C"),
            DataItem(
                "Question: What is an example of a raw material being something that comes directly from a source?\nChoices: A) cats from a shelter, B) birds from a nest, C) manure from a field, D) salt water from the ocean\nAnswer:",
                "Raw material refers to unprocessed material. Out of all the choices, only saltwater is unprocessed. So, the answer is D.",
                "D"),
            DataItem(
                "Question: If a road is bumpy and another road is smooth, the bumpier road\nChoices: A) will be lower than the smooth road, B) will be longer than the smooth road, C) will be rough on tires, D) will have cars driving quickly\nAnswer:",
                "A bumpier road causes higher damage to a vehicle, making it rough on tires. So, the answer is C.",
                "C"),
            DataItem(
                "Question: Revolution happens when\nChoices: A) Earth orbits the moon, B) Mars orbits the sun, C) the sun orbits the Earth, D) the Earth\'s moon orbits Mars\nAnswer:",
                "Orbital revolution is the movement of a planet around a star, or a moon around a planet. Out of all the choices, only Mars obits the sun is correct according to the definition. So, the answer is B.",
                "B"),
        ]
        return few_shot_examples

    @classmethod
    def get_task(cls, item: dict) -> DataItem:
        choices = ", ".join([f"{choice['label']}) {choice['text']}" for choice in item["question"]["choices"]])
        return DataItem(f"Question: {item['question']['stem']}\nChoices: {choices}\nAnswer:", None,
                        item["answerKey"])

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
        pattern = r"([A-D]\))"
        secondary_pattern = r"answer is ([A-D])"

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
        elif re.search(secondary_pattern, last_line) is not None:
            extracted_answer = re.search(secondary_pattern, last_line)
        elif re.search(pattern, first_line) is not None:
            extracted_answer = re.search(pattern, first_line)
        elif re.search(secondary_pattern, first_line) is not None:
            extracted_answer = re.search(secondary_pattern, first_line)
        else:
            extracted_answer = None

        if extracted_answer is not None:
            prediction = extracted_answer.group(1)[0]
            logger.debug(f"Prediction: {prediction}, Answer: {answer}")
            return prediction == answer, prediction
        logger.debug(f"Could not extract prediction from response")
        return False, ""

    def __str__(self) -> str:
        return "OpenBookQA"
