from src.models.data_item import DataItem
from src.prompting.prompting import Prompting


class NullShotV2(Prompting):
    """Null-Shot Prompting (Variant 2)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_prompt(cls, prompt: str, examples: list[DataItem] = None) -> str:
        magic = 'Look at examples in the "Examples" section and perform the following task.\n'
        return magic + prompt

    def __str__(self) -> str:
        return "Null-Shot Prompting (Variant 2)"
