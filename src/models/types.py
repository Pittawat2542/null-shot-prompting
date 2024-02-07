from enum import Enum


class LLMs(Enum):
    gpt_three_point_five_turbo = "gpt-3.5-turbo"
    gpt_four_turbo = "gpt-4-turbo"

    palm_two_text = "palm-2-text"
    palm_two_chat = "palm-2-chat"

    gemini_pro_text = "gemini-pro-text"
    gemini_pro_chat = "gemini-pro-chat"

    llama_two_seven = "llama-2-7b"
    llama_two_thirteen = "llama-2-13b"
    llama_two_seventy = "llama-2-70b"

    llama_two_chat_seven = "llama-2-chat-7b"
    llama_two_chat_thirteen = "llama-2-chat-13b"
    llama_two_chat_seventy = "llama-2-chat-70b"

    pythia_14m = "pythia-14m"
    pythia_31m = "pythia-31m"
    pythia_70m = "pythia-70m"
    pythia_160m = "pythia-160m"
    python_410m = "python-410m"
    pythia_1b = "pythia-1b"
    pythia_1_4b = "pythia-1.4b"
    pythia_2_8b = "pythia-2.8b"
    pythia_6_9b = "pythia-6.9b"
    pythia_12b = "pythia-12b"

    qwen_1_5_500m_chat = "qwen-1.5-0.5b-chat"
    qwen_1_5_1_8b_chat = "qwen-1.5-1.8b-chat"
    qwen_1_5_4b_chat = "qwen-1.5-4b-chat"
    qwen_1_5_7b_chat = "qwen-1.5-7b-chat"
    qwen_1_5_14b_chat = "qwen-1.5-14b-chat"
    qwen_1_5_72b_chat = "qwen-1.5-72b-chat"

    all = "all"


class Tasks(Enum):
    all = "all"
    # Closed-book QA
    TriviaQA = "triviaqa"
    # NLI
    ANLI = "anli"
    # Translation
    WMTJAEN = "wmt-ja-en"
    WMTENJA = "wmt-en-ja"
    # Reading comprehension
    RACE_H = "race-h"
    RACE_M = "race-m"
    # Winogrande
    Winogrande = "winogrande"
    # Commonsense reasoning
    CommonsenseQA = "csqa"
    StrategyQA = "strategyqa"
    OpenBookQA = "openbookqa"
    # Arithmetic reasoning
    AQuA = "aqua"
    GSM8K = "gsm8k"
    SVAMP = "svamp"


class Prompting(Enum):
    zero_shot = "zero-shot"
    few_shot = "few-shot"
    null_shot = "null-shot"
    chain_of_thought = "cot"
    zero_shot_chain_of_thought = "zero-shot-cot"
    null_shot_chain_of_thought = "null-shot-cot"

    null_shot_after = "null-shot-after"
    null_shot_v1 = "null-shot-v1"
    null_shot_v2 = "null-shot-v2"
    null_shot_v3 = "null-shot-v3"

    all = "all"
