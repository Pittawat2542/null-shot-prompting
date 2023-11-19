from openai import Client
from transformers import Pipeline

from src.config import GPT_THREE_POINT_FIVE_TURBO_MODEL, GPT_FOUR_TURBO_MODEL, PALM_TWO_TEXT_MODEL, PALM_TWO_CHAT_MODEL, \
    LLAMA_TWO_SEVEN_MODEL, LLAMA_TWO_THIRTEEN_MODEL, LLAMA_TWO_SEVENTY_MODEL, LLAMA_TWO_CHAT_SEVEN_MODEL, \
    LLAMA_TWO_CHAT_THIRTEEN_MODEL, LLAMA_TWO_CHAT_SEVENTY_MODEL
from src.llms.gpt import GPT
from src.llms.llama_two import LlamaTwo
from src.llms.llama_two_chat import LlamaTwoChat
from src.llms.palm_two_chat import PaLMTwoChat
from src.llms.palm_two_text import PaLMTwoText
from src.models.types import LLMs, Tasks, Prompting
from src.prompting.chain_of_thought import ChainOfThought
from src.prompting.few_shot import FewShot
from src.prompting.null_shot import NullShot
from src.prompting.null_shot_chain_of_thought import NullShotChainOfThought
from src.prompting.zero_shot import ZeroShot
from src.prompting.zero_shot_chain_of_thought import ZeroShotChainOfThought
from src.tasks.anli import ANLI
from src.tasks.aqua import AQuA
from src.tasks.commonsense_qa import CommonsenseQA
from src.tasks.gsm8k import GSM8K
from src.tasks.open_book_qa import OpenBookQA
from src.tasks.strategyqa import StrategyQA
from src.tasks.svamp import SVAMP
from src.tasks.winogrande import Winogrande


def get_prompting(prompting: Prompting):
    match prompting:
        case Prompting.zero_shot:
            return ZeroShot
        case Prompting.few_shot:
            return FewShot
        case Prompting.null_shot:
            return NullShot
        case Prompting.chain_of_thought:
            return ChainOfThought
        case Prompting.zero_shot_chain_of_thought:
            return ZeroShotChainOfThought
        case Prompting.null_shot_chain_of_thought:
            return NullShotChainOfThought
        case _:
            raise NotImplementedError(f"Prompting {prompting.value} not implemented")


def get_model(model: LLMs, client: Client | Pipeline = None):
    match model:
        case LLMs.gpt_three_point_five_turbo | LLMs.gpt_four_turbo:
            return GPT(client)
        case LLMs.palm_two_text:
            return PaLMTwoText()
        case LLMs.palm_two_chat:
            return PaLMTwoChat()
        case LLMs.llama_two_seven | LLMs.llama_two_thirteen | LLMs.llama_two_seventy:
            return LlamaTwo(client)
        case LLMs.llama_two_chat_seven | LLMs.llama_two_chat_thirteen | LLMs.llama_two_chat_seventy:
            return LlamaTwoChat(client)
        case _:
            raise NotImplementedError(f"Model {model.value} not implemented")


def get_model_name(model: LLMs):
    match model:
        case LLMs.gpt_three_point_five_turbo:
            return GPT_THREE_POINT_FIVE_TURBO_MODEL
        case LLMs.gpt_four_turbo:
            return GPT_FOUR_TURBO_MODEL
        case LLMs.palm_two_text:
            return PALM_TWO_TEXT_MODEL
        case LLMs.palm_two_chat:
            return PALM_TWO_CHAT_MODEL
        case LLMs.llama_two_seven:
            return LLAMA_TWO_SEVEN_MODEL
        case LLMs.llama_two_thirteen:
            return LLAMA_TWO_THIRTEEN_MODEL
        case LLMs.llama_two_seventy:
            return LLAMA_TWO_SEVENTY_MODEL
        case LLMs.llama_two_chat_seven:
            return LLAMA_TWO_CHAT_SEVEN_MODEL
        case LLMs.llama_two_chat_thirteen:
            return LLAMA_TWO_CHAT_THIRTEEN_MODEL
        case LLMs.llama_two_chat_seventy:
            return LLAMA_TWO_CHAT_SEVENTY_MODEL
        case _:
            raise NotImplementedError(f"Model {model.value} not implemented")


def get_task(task: Tasks):
    match task:
        case Tasks.ANLI:
            return ANLI
        case Tasks.Winogrande:
            return Winogrande
        case Tasks.CommonsenseQA:
            return CommonsenseQA
        case Tasks.StrategyQA:
            return StrategyQA
        case Tasks.OpenBookQA:
            return OpenBookQA
        case Tasks.AQuA:
            return AQuA
        case Tasks.GSM8K:
            return GSM8K
        case Tasks.SVAMP:
            return SVAMP
        case _:
            raise NotImplementedError(f"Task {task.value} not implemented")
