from src.models.data_item import DataItem
from src.prompting.prompting import Prompting


class ChainOfThought(Prompting):
    """Chain-of-Thought Prompting."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_prompt(cls, prompt: str, examples: list[DataItem] = None) -> str:
        if examples is None:
            raise ValueError("Chain-of-Thought Prompting requires examples to be passed")
        if examples[0].rationale is None:
            raise ValueError("Chain-of-Thought Prompting requires examples to have rationales")
        cot_examples = ""
        for example in examples:
            cot_examples += f"Question: {example.prompt} {example.rationale}\n"
        return cot_examples + prompt

    def __str__(self) -> str:
        return "Chain-of-Thought Prompting"
