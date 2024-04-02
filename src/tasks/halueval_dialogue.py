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


class HaluEvalDialogue(Task, ABC):
    """HaluEval Dialogue"""

    dataset_path = Path(os.getcwd()) / DATASETS_DIRECTORY / 'HaluEval_dialogue.json'
    dev_dataset_path = Path(os.getcwd()) / DATASETS_DIRECTORY / 'HaluEval_dialogue.json'

    def __new__(cls, *args, **kwargs):
        if cls is HaluEvalDialogue:
            raise TypeError(f"'{cls.__name__}' cannot be instantiated")
        return object.__new__(cls)

    @classmethod
    def has_native_cot_samples_supported(cls) -> bool:
        return False

    @classmethod
    def get_few_shot_samples(cls) -> list[DataItem]:
        preamble = """You are a response judge. You MUST determine if the provided response contains non-factual or hallucinated information. The answer you give MUST be "Yes" or No"
- You are trying to determine if the true entity in the response is replaced with a highly similar entity.
- You are trying to determine if the true entity in the response is replaced with a dissimilar entity.
- You are trying to determine if the true entity in the response is replaced with a dissimilar entity in a different entity type.
I want you act as a response judge. Given a dialogue history and a response, your objective is to determine if the provided response contains non-factual or hallucinated information. You SHOULD give your judgement based on the following hallucination types and the world knowledge.
You should try your best to determine if the response contains non-factual or hallucinated information according to the above hallucination types. The answer you give MUST be "Yes" or "No"."""
        return [DataItem(f"""{preamble}
#Dialogue History#: [Human]: Could you recommand movies similar to The Dark Knight? [Assistant]: The sequel to Batman Begins is The Dark Knight. [Human]: Okay. Who is the director of The Dark Knight and any other movies from him not related to Batman?
#Response#: Christopher Nolan was the director. He also directed insomnia and inception.
#Your Judgement#:""", None, "B"), DataItem(f"""{preamble}
#Dialogue History#: [Human]: Could you recommand movies similar to The Dark Knight? [Assistant]: The sequel to Batman Begins is The Dark Knight. [Human]: Okay. Who is the director of The Dark Knight and any other movies from him not related to Batman?
#Response#: Steven Spielberg was the director. He also directed insomnia and inception.
#Your Judgement#:""", None, "A"), DataItem(f"""{preamble}
#Dialogue History#: [Human]: Could you recommand movies similar to The Dark Knight? [Assistant]: The sequel to Batman Begins is The Dark Knight. [Human]: Okay. Who is the director of The Dark Knight and any other movies from him not related to Batman?
#Response#: Christopher Nolan was the director. He also directed insomnia and inception.
#Your Judgement#:""", None, "B"), DataItem(f"""{preamble}
#Dialogue History#: [Human]: Could you recommand movies similar to The Dark Knight? [Assistant]: The sequel to Batman Begins is The Dark Knight. [Human]: Okay. Who is the director of The Dark Knight and any other movies from him not related to Batman?
#Response#: Batman Begins was the director. He also directed insomnia and inception.
#Your Judgement#:""", None, "A"), DataItem(f"""{preamble}
#Dialogue History#: [Human]: Could you recommand movies similar to The Dark Knight? [Assistant]: The sequel to Batman Begins is The Dark Knight. [Human]: Okay. Who is the director of The Dark Knight and any other movies from him not related to Batman?
#Response#: Christopher Nolan was the director. He also directed insomnia and inception.
#Your Judgement#:""", None, "B"), DataItem(f"""{preamble}
#Dialogue History#: [Human]: Could you recommand movies similar to The Dark Knight? [Assistant]: The sequel to Batman Begins is The Dark Knight. [Human]: Okay. Who is the director of The Dark Knight and any other movies from him not related to Batman?
#Response#: United States of America was the director. He also directed insomnia and inception.
#Your Judgement#:""", None, "A")]

    @classmethod
    def get_task(cls, item: dict) -> DataItem:
        response_type = random.choice(["right_response", "hallucinated_response"])
        if response_type == "right_response":
            label = "B"
        else:
            label = "A"
        prompt = f"""You are a response judge. You MUST determine if the provided response contains non-factual or hallucinated information. The answer you give MUST be "Yes" or No"
- You are trying to determine if the true entity in the response is replaced with a highly similar entity.
- You are trying to determine if the true entity in the response is replaced with a dissimilar entity.
- You are trying to determine if the true entity in the response is replaced with a dissimilar entity in a different entity type.
I want you act as a response judge. Given a dialogue history and a response, your objective is to determine if the provided response contains non-factual or hallucinated information. You SHOULD give your judgement based on the following hallucination types and the world knowledge.
You should try your best to determine if the response contains non-factual or hallucinated information according to the above hallucination types. The answer you give MUST be "Yes" or "No".
#Dialogue History#: {item['dialogue_history']}
#Response#: {item[response_type]}
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
        return "HaluEvalDialogue"
