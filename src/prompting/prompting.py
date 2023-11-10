from abc import ABC, abstractmethod

from src.models.data_item import DataItem


class Prompting(ABC):
    """Abstract class for Prompting."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    @abstractmethod
    def get_prompt(cls, prompt: str, examples: list[DataItem] = None) -> str:
        return ""

    @abstractmethod
    def __str__(self) -> str:
        return ""
