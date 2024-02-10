from time import perf_counter, sleep

import google.generativeai as genai
from google.api_core.exceptions import InvalidArgument, ServiceUnavailable, InternalServerError, TooManyRequests, \
    DeadlineExceeded
from google.generativeai.types.generation_types import StopCandidateException, BlockedPromptException
from loguru import logger

from src.config import GEMINI_RATE_LIMIT
from src.llms.llm import LLM


class GeminiProText(LLM):
    """Gemini Pro Text"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def inference(self, prompt: str, model_name="") -> (str, dict):
        model = genai.GenerativeModel('gemini-pro')
        logger.info("Generating response from Gemini Pro Text")
        start_time = perf_counter()
        try:
            text_completion = model.generate_content(prompt,
                                                     generation_config=genai.types.GenerationConfig(temperature=0))
        except (ServiceUnavailable, InternalServerError, TooManyRequests, DeadlineExceeded):
            sleep(5)
            text_completion = model.generate_content(prompt,
                                                     generation_config=genai.types.GenerationConfig(temperature=0))
        except (StopCandidateException, BlockedPromptException) as e:
            logger.debug(f"Error: {e}")
            text_completion = lambda: None
            text_completion.text = ""
        except InvalidArgument as e:
            if "The requested language is not supported" in str(e):
                text_completion = lambda: None
                text_completion.text = f"ERROR: {e}"
            else:
                raise e
        end_time = perf_counter()

        try:
            if len(text_completion.parts) == 0:
                logger.debug("No response generated, returning empty string")
                response = ""
            else:
                response = text_completion.text
        except ValueError as e:
            logger.debug(f"Error: {e}\n{text_completion.prompt_feedback}")
            response = ""
        except AttributeError as e:
            logger.debug(f"Error: {e}")
            response = ""

        logger.debug(response)
        logger.success(
            f"Response generated from Gemini Pro Text, response length: {len(response)}, "
            f"time taken: {end_time - start_time} seconds")
        if end_time - start_time < GEMINI_RATE_LIMIT:
            sleep(GEMINI_RATE_LIMIT - (end_time - start_time))
        return response, {"length": len(response), "time_taken": end_time - start_time, "start_time": start_time,
                          "end_time": end_time}

    def __str__(self) -> str:
        return "Gemini Pro Text"
