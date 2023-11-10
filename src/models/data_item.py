class DataItem:
    prompt: str
    rationale: str | None
    label: str

    def __init__(self, prompt: str, rationale: str | None, label: str):
        self.prompt = prompt
        self.rationale = rationale
        self.label = label
