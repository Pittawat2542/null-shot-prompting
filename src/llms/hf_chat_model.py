from time import perf_counter

from loguru import logger
from transformers import Conversation, Pipeline

from src.llms.llm import LLM


class HuggingFaceChatModel(LLM):
    """HuggingFace chat generative model"""

    def __init__(self, client: Pipeline, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = client

    def inference(self, prompt: str, model_name="") -> (str, dict):
        logger.info(f"Generating response from {model_name}")
        start_time = perf_counter()
        conversation = Conversation(prompt)
        response = self.client(conversation, do_sample=False).generated_responses[-1]
        end_time = perf_counter()
        logger.debug(response)
        logger.success(
            f"Response generated from {model_name}, response length: {len(response)}, "
            f"time taken: {end_time - start_time} seconds")
        return response, {"length": len(response), "time_taken": end_time - start_time, "start_time": start_time,
                          "end_time": end_time}

    def __str__(self) -> str:
        return "HuggingFace chat generative model"
