from src.models.data_item import DataItem
from src.prompting.prompting import Prompting


class NullShotV3(Prompting):
    """Null-Shot Prompting (Variant 3)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_prompt(cls, prompt: str, examples: list[DataItem] = None) -> str:
        magic = 'Perform the following task as demonstrated in the "Examples" section.\n'
        return magic + prompt

    def __str__(self) -> str:
        return "Null-Shot Prompting (Variant 3)"
