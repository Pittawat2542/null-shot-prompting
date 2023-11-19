from time import perf_counter

from loguru import logger
from transformers import Pipeline

from src.llms.llm import LLM


class LlamaTwo(LLM):
    """Llama 2 model"""

    def __init__(self, client: Pipeline, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = client

    def inference(self, prompt: str, model_name="") -> (str, dict):
        logger.info(f"Generating response from Llama 2 {model_name}")
        start_time = perf_counter()
        text_completion = self.client(prompt, do_sample=False)
        end_time = perf_counter()
        response = text_completion[-1]['generated_text'].replace(prompt, '')
        logger.debug(response)
        logger.success(
            f"Response generated from Llama 2 {model_name}, response length: {len(response)}, "
            f"time taken: {end_time - start_time} seconds")
        return response, {"length": len(response), "time_taken": end_time - start_time, "start_time": start_time,
                          "end_time": end_time}

    def __str__(self) -> str:
        return "Llama 2"
