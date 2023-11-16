from abc import ABC, abstractmethod

from src.models.data_item import DataItem


class Task(ABC):
    """Abstract class for Task."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    @abstractmethod
    def has_native_cot_samples_supported(cls) -> bool:
        return False

    @classmethod
    @abstractmethod
    def get_few_shot_samples(cls) -> list[DataItem]:
        return []

    @classmethod
    @abstractmethod
    def get_task(cls, item: dict) -> DataItem:
        return DataItem("", "", "")

    @classmethod
    @abstractmethod
    def get_task_list(cls) -> list[DataItem]:
        return []

    @classmethod
    @abstractmethod
    def evaluate(cls, response: str, answer: str) -> (bool, str):
        return False, ""

    @abstractmethod
    def __str__(self) -> str:
        return ""
