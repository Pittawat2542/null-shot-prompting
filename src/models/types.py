from enum import Enum


class LLMs(Enum):
    gpt_three_point_five_turbo = "gpt-3.5-turbo"
    gpt_four_turbo = "gpt-4-turbo"
    palm_two_text = "palm-2-text"
    palm_two_chat = "palm-2-chat"
    llama_two_seven = "llama-2-7b"
    llama_two_thirteen = "llama-2-13b"
    llama_two_seventy = "llama-2-70b"
    llama_two_chat_seven = "llama-2-chat-7b"
    llama_two_chat_thirteen = "llama-2-chat-13b"
    llama_two_chat_seventy = "llama-2-chat-70b"
    all = "all"


class Tasks(Enum):
    # Commonsense reasoning
    CommonsenseQA = "csqa"
    StrategyQA = "strategyqa"
    OpenBookQA = "openbookqa"
    # Arithmetic reasoning
    AQuA = "aqua"
    GSM8K = "gsm8k"
    all = "all"


class Prompting(Enum):
    zero_shot = "zero-shot"
    few_shot = "few-shot"
    null_shot = "null-shot"
    chain_of_thought = "cot"
    zero_shot_chain_of_thought = "zero-shot-cot"
    null_shot_chain_of_thought = "null-shot-cot"
    all = "all"
