from time import perf_counter, sleep

from anthropic import APITimeoutError, APIConnectionError, APIStatusError, Anthropic
from loguru import logger

from src.config import CLAUDE_RATE_LIMIT
from src.llms.llm import LLM


class Claude(LLM):
    """Claude"""

    def __init__(self, client: Anthropic, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = client

    def inference(self, prompt: str, model_name="") -> (str, dict):
        logger.info(f"Generating response from {model_name}")
        start_time = perf_counter()
        try:
            chat_completion = self.client.messages.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=4096
            )
            end_time = perf_counter()
            response = chat_completion.content[0].text
        except (APITimeoutError, APIConnectionError, APIStatusError) as e:
            print(e)
            raise e
        except Exception as e:
            end_time = perf_counter()
            response = f"ERROR: {e}"
        logger.debug(response)
        logger.success(
            f"Response generated from {model_name}, response length: {len(response)}, "
            f"time taken: {end_time - start_time} seconds")
        if end_time - start_time < CLAUDE_RATE_LIMIT:
            sleep(CLAUDE_RATE_LIMIT - (end_time - start_time))
        return response, {"length": len(response), "time_taken": end_time - start_time, "start_time": start_time,
                          "end_time": end_time}

    def __str__(self) -> str:
        return "Claude"
