from src.models.data_item import DataItem
from src.prompting.prompting import Prompting


class FewShot(Prompting):
    """Few-Shot Prompting."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_prompt(cls, prompt: str, examples: list[DataItem] = None) -> str:
        if examples is None:
            raise ValueError("Few-Shot Prompting requires examples to be passed")
        few_shot_examples = ""
        for example in examples:
            few_shot_examples += f"Question: {example.prompt} {example.label}\n"
        return few_shot_examples + prompt

    def __str__(self) -> str:
        return "Few-Shot Prompting"
