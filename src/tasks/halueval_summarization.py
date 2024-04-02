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


class HaluEvalSummarization(Task, ABC):
    """HaluEval Summarization"""

    dataset_path = Path(os.getcwd()) / DATASETS_DIRECTORY / 'HaluEval_summarization.json'
    dev_dataset_path = Path(os.getcwd()) / DATASETS_DIRECTORY / 'HaluEval_summarization.json'

    def __new__(cls, *args, **kwargs):
        if cls is HaluEvalSummarization:
            raise TypeError(f"'{cls.__name__}' cannot be instantiated")
        return object.__new__(cls)

    @classmethod
    def has_native_cot_samples_supported(cls) -> bool:
        return False

    @classmethod
    def get_few_shot_samples(cls) -> list[DataItem]:
        preamble = """I want you act as a summary judge. Given a document and a summary, your objective is to determine if the provided summary contains non-factual or hallucinated information. You SHOULD give your judgement based on the following hallucination types and the world knowledge.
- You are trying to determine if the summary is factual but some information cannot be directly inferred or entailed from the document.
- You are trying to determine if there exists some non-factual and incorrect information in the summary.
- You are trying to determine if there is a factual contradiction between the summary and the document.
You should try your best to determine if the summary contains non-factual or hallucinated information according to the above hallucination types. The answer you give MUST be "Yes" or "No"."""
        return [DataItem(f"""{preamble}
#Document#: The panther chameleon was found on Monday by a dog walker in the wooded area at Marl Park. It had to be put down after X-rays showed all of its legs were broken and it had a deformed spine. RSPCA Cymru said it was an "extremely sad example of an abandoned and neglected exotic pet". Inspector Selina Chan said: "It is a possibility that the owners took on this animal but were unable to provide the care he needs and decided to release him to the wild. "We are urging potential owners of exotic animals to thoroughly research what is required in the care of the particular species before taking one on. "Potential owners need to make sure they can give their animal the environment it needs and they have the facilities, time, financial means and long-term commitment to maintain a good standard of care, as required under the Animal Welfare Act 2006." She added it was illegal to release non-native species into the wild.
#Summary#: A chameleon that was found in a Cardiff park has been put down after being abandoned and neglected by its owners.
#Your Judgement#:""", None, "A"), DataItem(f"""{preamble}
#Document#: The city was brought to a standstill on 15 December last year when a gunman held 18 hostages for 17 hours. Family members of victims Tori Johnson and Katrina Dawson were in attendance. Images of the floral tributes that filled the city centre in the wake of the siege were projected on to the cafe and surrounding buildings in an emotional twilight ceremony. Prime Minister Malcolm Turnbull gave an address saying a "whole nation resolved to answer hatred with love". "Testament to the spirit of Australians is that with such unnecessary, thoughtless tragedy, an amazing birth of mateship, unity and love occurs. Proud to be Australian," he said. How the Sydney siege unfolded New South Wales Premier Mike Baird has also announced plans for a permanent memorial to be built into the pavement in Martin Place. Clear cubes containing flowers will be embedded into the concrete and will shine with specialised lighting. It is a project inspired by the massive floral tributes that were left in the days after the siege. "Something remarkable happened here. As a city we were drawn to Martin Place. We came in shock and in sorrow but every step we took was with purpose," he said on Tuesday.
#Summary#: Crowds have gathered in Sydney's Martin Place to honour the victims of the Lindt cafe siege, one year on.
#Your Judgement#:""", None, "B"), DataItem(f"""{preamble}
#Document#: Christopher Huxtable, 34, from Swansea, had been missing since the collapse in February. His body was found on Wednesday and workers who carried out the search formed a guard of honour as it was driven from the site in the early hours of the morning. Ken Cresswell, 57, and John Shaw, 61, both from Rotherham, remain missing. The body of a fourth man, Michael Collings, 53, from Brotton, Teesside, was previously recovered from the site. Swansea East MP Carolyn Harris, who has been involved with the family since the incident, said they still did not know all the facts about the collapse. She said: "I feel very sad. My heart and my prayers go out to the family who have waited desperately for Christopher's body to be found. They can finally have closure, and say goodbye to him and grieve his loss. "But let's not forget that there's two other families who are still waiting for their loved ones to be returned." The building was due for demolition when it partially collapsed in February.
#Summary#: The body of a man whose body was found at the site of the Swansea Bay Power Station collapse has been removed from the site.
#Your Judgement#:""", None, "A")]

    @classmethod
    def get_task(cls, item: dict) -> DataItem:
        response_type = random.choice(["right_summary", "hallucinated_summary"])
        if response_type == "right_summary":
            label = "B"
        else:
            label = "A"
        prompt = f"""I want you act as a summary judge. Given a document and a summary, your objective is to determine if the provided summary contains non-factual or hallucinated information. You SHOULD give your judgement based on the following hallucination types and the world knowledge.
- You are trying to determine if the summary is factual but some information cannot be directly inferred or entailed from the document.
- You are trying to determine if there exists some non-factual and incorrect information in the summary.
- You are trying to determine if there is a factual contradiction between the summary and the document.
You should try your best to determine if the summary contains non-factual or hallucinated information according to the above hallucination types. The answer you give MUST be "Yes" or "No".
#Document#: {item['document']}
#Summary#: {item[response_type]}
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
        return "HaluEvalSummarization"
