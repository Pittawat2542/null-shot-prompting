from time import perf_counter, sleep

import google.generativeai as genai
from google.api_core.exceptions import InvalidArgument, ServiceUnavailable, InternalServerError, TooManyRequests
from loguru import logger

from src.config import GEMINI_RATE_LIMIT
from src.llms.llm import LLM


class GeminiProChat(LLM):
    """Gemini Pro Chat"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def inference(self, prompt: str, model_name="") -> (str, dict):
        model = genai.GenerativeModel('gemini-pro')
        chat = model.start_chat()
        logger.info("Generating response from Gemini Pro Chat")
        start_time = perf_counter()
        try:
            chat_completion = chat.send_message(prompt, generation_config=genai.types.GenerationConfig(temperature=0))
        except (ServiceUnavailable, InternalServerError, TooManyRequests):
            sleep(5)
            chat_completion = chat.send_message(prompt, generation_config=genai.types.GenerationConfig(temperature=0))
        except InvalidArgument as e:
            if "The requested language is not supported" in str(e):
                chat_completion = lambda: None
                chat_completion.text = f"ERROR: {e}"
            else:
                raise e
        end_time = perf_counter()

        try:
            if len(chat_completion.parts) == 0:
                logger.debug("No response generated, returning empty string")
                response = ""
            else:
                response = chat_completion.text
        except ValueError as e:
            logger.debug(f"Error: {e}\n{chat_completion.prompt_feedback}")
            response = ""

        logger.debug(response)
        logger.success(
            f"Response generated from Gemini Pro Chat, response length: {len(response)}, "
            f"time taken: {end_time - start_time} seconds")
        if end_time - start_time < GEMINI_RATE_LIMIT:
            sleep(GEMINI_RATE_LIMIT - (end_time - start_time))
        return response, {"length": len(response), "time_taken": end_time - start_time, "start_time": start_time,
                          "end_time": end_time}

    def __str__(self) -> str:
        return "Gemini Pro Chat"
