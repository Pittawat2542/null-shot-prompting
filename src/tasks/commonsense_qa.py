import os
import re
from abc import ABC
from pathlib import Path

from jsonlines import jsonlines
from loguru import logger

from src.config import DATASETS_DIRECTORY
from src.models.data_item import DataItem
from src.tasks.task import Task


class CommonsenseQA(Task, ABC):
    """CommonsenseQA"""

    dataset_path = Path(os.getcwd()) / DATASETS_DIRECTORY / 'CommonsenseQA.jsonl'
    dev_dataset_path = Path(os.getcwd()) / DATASETS_DIRECTORY / 'CommonsenseQA_dev.jsonl'

    def __new__(cls, *args, **kwargs):
        if cls is CommonsenseQA:
            raise TypeError(f"'{cls.__name__}' cannot be instantiated")
        return object.__new__(cls)

    @classmethod
    def has_native_cot_samples_supported(cls) -> bool:
        return False

    @classmethod
    def get_few_shot_samples(cls) -> list[DataItem]:
        # From CoT paper
        few_shot_examples = [
            DataItem(
                "Question: What do people use to absorb extra ink from a fountain pen?\nChoices: A) shirt pocket, B) calligrapher’s hand, C) inkwell, D) desk drawer, E) blotter\nAnswer:",
                "The answer must be an item that can absorb ink. Of the above choices, only blotters are used to absorb ink. So, the answer is E.",
                "E"),
            DataItem(
                "Question: What home entertainment equipment requires cable?\nChoices: A) radio shack, B) substation, C) television, D) cabinet\nAnswer:",
                "The answer must require cable. Of the above choices, only television requires cable. So, the answer is C.",
                "C"),
            DataItem(
                "Question: The fox walked from the city into the forest, what was it looking for?\nChoices: A) pretty flowers, B) hen house, C) natural habitat, D) storybook\nAnswer:",
                "The answer must be something in the forest. Of the above choices, only natural habitat is in the forest. So, the answer is B.",
                "B"),
            DataItem(
                "Question: Sammy wanted to go to where the people were. Where might he go?\nChoices: A) populated areas, B) race track, C) desert, D) apartment, E) roadblock\nAnswer:",
                "The answer must be a place with a lot of people. Of the above choices, only populated areas have a lot of people. So, the answer is A.",
                "A"),
            DataItem(
                "Question: Where do you put your grapes just before checking out?\nChoices: A) mouth, B) grocery cart, C)super market, D) fruit basket, E) fruit market\nAnswer:",
                "The answer should be the place where grocery items are placed before checking out. Of the above choices, grocery cart makes the most sense for holding grocery items. So, the answer is B.",
                "B"),
            DataItem(
                "Question: Google Maps and other highway and street GPS services have replaced what?\nChoices: A) united states, B) mexico, C) countryside, D) atlas\nAnswer:",
                "The answer must be something that used to do what Google Maps and GPS services do, which is to give directions. Of the above choices, only atlases are used to give directions. So, the answer is D.",
                "D"),
            DataItem(
                "Question: Before getting a divorce, what did the wife feel who was doing all the work?\nChoices: A) harder, B) anguish, C) bitterness, D) tears, E) sadness\nAnswer:",
                "The answer should be the feeling of someone getting divorced who was doing all the work. Of the above choices, the closest feeling is bitterness. So, the answer is C.",
                "C")
        ]
        return few_shot_examples

    @classmethod
    def get_task(cls, item: dict) -> DataItem:
        choices = ", ".join([f"{choice['label']}) {choice['text']}" for choice in item["question"]["choices"]])
        return DataItem(f"Question: {item['question']['stem']}\nChoices: {choices}\nAnswer:", None,
                        item.get("answerKey", "-"))

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
        return "CommonsenseQA"
