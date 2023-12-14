from src.models.data_item import DataItem
from src.prompting.prompting import Prompting


class NullShotV1(Prompting):
    """Null-Shot Prompting (Variant 1)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_prompt(cls, prompt: str, examples: list[DataItem] = None) -> str:
        magic = 'Utilize examples and information from the "Examples" section to perform the following task.\n'
        return magic + prompt

    def __str__(self) -> str:
        return "Null-Shot Prompting (Variant 1)"
