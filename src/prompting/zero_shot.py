from src.models.data_item import DataItem
from src.prompting.prompting import Prompting


class ZeroShot(Prompting):
    """Zero-Shot Prompting."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_prompt(cls, prompt: str, examples: list[DataItem] = None) -> str:
        return prompt

    def __str__(self) -> str:
        return "Zero-Shot Prompting"
