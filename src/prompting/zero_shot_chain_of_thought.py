from src.models.data_item import DataItem
from src.prompting.prompting import Prompting


class ZeroShotChainOfThought(Prompting):
    """Zero-shot Chain-of-Thought Prompting."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_prompt(cls, prompt: str, examples: list[DataItem] = None) -> str:
        magic = "Let's think step-by-step."
        return prompt + magic

    def __str__(self) -> str:
        return "Zero-Shot Chain-of-Thought Prompting"
