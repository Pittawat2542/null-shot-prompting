from src.models.data_item import DataItem
from src.prompting.prompting import Prompting


class NullShot(Prompting):
    """Null-Shot Prompting."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_prompt(cls, prompt: str, examples: list[DataItem] = None) -> str:
        magic = ('Look at examples in the "Examples" section and utilize examples and information from that section to '
                 'perform the following task.\n')
        return magic + prompt

    def __str__(self) -> str:
        return "Null-Shot Prompting"
