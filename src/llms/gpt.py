from time import perf_counter, sleep

import openai
from loguru import logger

from src.config import RANDOM_SEED, OPENAI_RATE_LIMIT, GPT_THREE_POINT_FIVE_TURBO_MODEL, GPT_FOUR_TURBO_MODEL
from src.llms.llm import LLM


class GPT(LLM):
    """GPT"""

    def __init__(self, client: openai.Client, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = client

    def inference(self, prompt: str, model_name="") -> str:
        model = ""
        match model_name:
            case "gpt-3.5-turbo":
                model = GPT_THREE_POINT_FIVE_TURBO_MODEL
            case "gpt-4-turbo":
                model = GPT_FOUR_TURBO_MODEL
            case _:
                raise NotImplementedError(f"Model {model_name} not implemented")
        logger.info(f"Generating response from {model_name}")
        start_time = perf_counter()
        chat_completion = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            seed=RANDOM_SEED,
        )
        end_time = perf_counter()
        response = chat_completion.choices[0].message.content
        logger.debug(response)
        logger.success(
            f"Response generated from {model}, response length: {len(response)}, "
            f"time taken: {end_time - start_time} seconds")
        if end_time - start_time < OPENAI_RATE_LIMIT:
            sleep(OPENAI_RATE_LIMIT - (end_time - start_time))
        return response

    def __str__(self) -> str:
        return "GPT"
