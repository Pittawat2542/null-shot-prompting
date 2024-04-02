import os
import random
import re
from abc import ABC
from pathlib import Path

from jsonlines import jsonlines
from loguru import logger

from src.config import DATASETS_DIRECTORY
from src.models.data_item import DataItem
from src.tasks.task import Task


class HaluEvalQA(Task, ABC):
    """HaluEval QA"""

    dataset_path = Path(os.getcwd()) / DATASETS_DIRECTORY / 'HaluEval_qa.json'
    dev_dataset_path = Path(os.getcwd()) / DATASETS_DIRECTORY / 'HaluEval_qa.json'

    def __new__(cls, *args, **kwargs):
        if cls is HaluEvalQA:
            raise TypeError(f"'{cls.__name__}' cannot be instantiated")
        return object.__new__(cls)

    @classmethod
    def has_native_cot_samples_supported(cls) -> bool:
        return False

    @classmethod
    def get_few_shot_samples(cls) -> list[DataItem]:
        preamble = """I want you act as an answer judge. Given a question and an answer, your objective is to determine if the provided answer contains non-factual or hallucinated information. You SHOULD give your judgement based on the following hallucination types and the world knowledge.
- You are trying to determine if the answer misunderstands the question context and intention.
- You are trying to determine if there is a factual contradiction between the answer and the world knowledge. Some information in the answer might be fabricated.
- You are trying to determine if the answer is too general or too specific to answer the question at an appropriate level of specificity.
- You are trying to determine if the answer can be correctly inferred from the knowledge.
You should try your best to determine if the answer contains non-factual or hallucinated information according to the above hallucination types. The answer you give MUST be "Yes" or "No"."""
        return [DataItem(f"""{preamble}
#Question#: What is a rare breed of dog that was derived as a variant of Rat Terrier, Shiloh Shepherd dog or American Hairless Terrier?
#Answer#: American Hairless Terrier
#Your Judgement#:""", None, "B"), DataItem(f"""{preamble}
#Question#: Are the New Orleans Outfall Canals the same length as the Augusta Canal?
#Answer#: No, the New Orleans Outfall Canals and the Augusta Canal are not the same length. The Orleans Canal is approximately 3.6 miles (5.8 kilometers) long while the Augusta Canal is approximately 7 miles (11.3 kilometers) long.
#Your Judgement#:""", None, "A"), DataItem(f"""{preamble}
#Question#: What U.S Highway gives access to Zilpo Road, and is also known as Midland Trail?
#Answer#: U.S Highway 70
#Your Judgement#:""", None, "B"), DataItem(f"""{preamble}
#Question#: What genre do Superheaven and Oceansize belong to?
#Answer#: Superheaven and Oceansize belong to the rock genre.
#Your Judgement#:""", None, "B"), DataItem(f"""{preamble}
#Question#: What profession do Kōbō Abe and Agatha Christie share?
#Answer#: Playwright.
#Your Judgement#:""", None, "B"), DataItem(f"""{preamble}
#Question#: Which band has more members, Muse or The Raconteurs?
#Answer#: Muse has more members than The Raconteurs.
#Your Judgement#:""", None, "A"), DataItem(f"""{preamble}
#Question#: Which is currently more valuable, Temagami-Lorrain Mine or Meadowbank Gold Mine?
#Answer#: Meadowbank Gold Mine, since Meadowbank Gold Mine is still producing gold and the TemagamiLorrain Mine has been inactive for years.
#Your Judgement#:""", None, "B")]

    @classmethod
    def get_task(cls, item: dict) -> DataItem:
        response_type = random.choice(["right_answer", "hallucinated_answer"])
        if response_type == "right_answer":
            label = "B"
        else:
            label = "A"
        prompt = f"""I want you act as an answer judge. Given a question and an answer, your objective is to determine if the provided answer contains non-factual or hallucinated information. You SHOULD give your judgement based on the following hallucination types and the world knowledge.
- You are trying to determine if the answer misunderstands the question context and intention.
- You are trying to determine if there is a factual contradiction between the answer and the world knowledge. Some information in the answer might be fabricated.
- You are trying to determine if the answer is too general or too specific to answer the question at an appropriate level of specificity.
- You are trying to determine if the answer can be correctly inferred from the knowledge.
You should try your best to determine if the answer contains non-factual or hallucinated information according to the above hallucination types. The answer you give MUST be "Yes" or "No".
#Question: {item['question']}
#Answer#: {item[response_type]}
ํ#Your Judgement#:"""
        return DataItem(prompt, None, label)

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
        pattern = r"([A-B]\))|(Yes|No)"
        secondary_pattern = r"answer is ([A-B])|(Yes|No)"

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
            prediction = extracted_answer.group(1) if extracted_answer.group(
                1) is not None else extracted_answer.group(0)
            if prediction == "Yes":
                prediction = "A"
            elif prediction == "No":
                prediction = "B"
            else:
                prediction = prediction[0]
            logger.debug(f"Prediction: {prediction}, Answer: {answer}")
            return prediction == answer, prediction
        logger.debug(f"Could not extract prediction from response")
        return False, ""

    def __str__(self) -> str:
        return "HaluEvalQA"
