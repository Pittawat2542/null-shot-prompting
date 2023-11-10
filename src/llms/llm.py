from abc import ABC, abstractmethod


class LLM(ABC):
    """Abstract class for LLMs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def inference(self, prompt: str) -> str:
        return ""

    @abstractmethod
    def __str__(self) -> str:
        return ""
