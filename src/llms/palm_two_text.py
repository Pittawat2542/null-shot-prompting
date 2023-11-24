from time import perf_counter, sleep

import google.generativeai as palm
from google.api_core.exceptions import ServiceUnavailable, InvalidArgument
from loguru import logger

from src.config import PALM_RATE_LIMIT
from src.llms.llm import LLM


class PaLMTwoText(LLM):
    """PaLM 2 Text"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def inference(self, prompt: str, model_name="") -> (str, dict):
        logger.info("Generating response from PaLM 2 Text")
        start_time = perf_counter()
        try:
            chat_completion = palm.generate_text(prompt=prompt, temperature=0)
        except ServiceUnavailable:
            chat_completion = palm.generate_text(prompt=prompt, temperature=0)
        except InvalidArgument as e:
            if "The requested language is not supported" in str(e):
                chat_completion = f"ERROR: {e}"
            else:
                raise e
        end_time = perf_counter()
        response = chat_completion.result
        if response is None:
            logger.debug("No response generated, returning empty string")
            response = ""
        logger.debug(response)
        logger.success(
            f"Response generated from PaLM 2 Text, response length: {len(response)}, "
            f"time taken: {end_time - start_time} seconds")
        if end_time - start_time < PALM_RATE_LIMIT:
            sleep(PALM_RATE_LIMIT - (end_time - start_time))
        return response, {"length": len(response), "time_taken": end_time - start_time, "start_time": start_time,
                          "end_time": end_time}

    def __str__(self) -> str:
        return "PaLM 2 Text"
