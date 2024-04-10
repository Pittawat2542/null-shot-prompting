from time import perf_counter, sleep

from anthropic import APITimeoutError, APIConnectionError, APIStatusError, Anthropic, BadRequestError
from loguru import logger

from src.config import CLAUDE_RATE_LIMIT, CLAUDE_2_1_MODEL, CLAUDE_3_HAIKU_MODEL, CLAUDE_3_SONNET_MODEL, \
    CLAUDE_3_OPUS_MODEL
from src.llms.llm import LLM


class Claude(LLM):
    """Claude"""

    def __init__(self, client: Anthropic, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = client

    def inference(self, prompt: str, model_name="") -> (str, dict):
        model = ""
        match model_name:
            case "claude-2.1":
                model = CLAUDE_2_1_MODEL
            case "claude-3-haiku":
                model = CLAUDE_3_HAIKU_MODEL
            case "claude-3-sonnet":
                model = CLAUDE_3_SONNET_MODEL
            case "claude-3-opus":
                model = CLAUDE_3_OPUS_MODEL
            case _:
                raise NotImplementedError(f"Model {model_name} not implemented")

        logger.info(f"Generating response from {model}")
        start_time = perf_counter()
        try:
            chat_completion = self.client.messages.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=4096
            )
            end_time = perf_counter()
            response = chat_completion.content[0].text
        except BadRequestError as e:
            logger.debug(f"Error: {e}")
            response = f"ERROR: {e}"
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
