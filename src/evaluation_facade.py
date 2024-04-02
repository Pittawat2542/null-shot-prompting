from anthropic import Anthropic
from openai import Client
from transformers import Pipeline

from src.config import (
    GPT_THREE_POINT_FIVE_TURBO_MODEL,
    GPT_FOUR_TURBO_MODEL,
    PALM_TWO_TEXT_MODEL,
    PALM_TWO_CHAT_MODEL,
    LLAMA_TWO_SEVEN_MODEL,
    LLAMA_TWO_THIRTEEN_MODEL,
    LLAMA_TWO_SEVENTY_MODEL,
    LLAMA_TWO_CHAT_SEVEN_MODEL,
    LLAMA_TWO_CHAT_THIRTEEN_MODEL,
    LLAMA_TWO_CHAT_SEVENTY_MODEL,
    GEMINI_PRO_TEXT_MODEL,
    GEMINI_PRO_CHAT_MODEL,
    PYTHIA_14M_MODEL,
    PYTHIA_1B_MODEL,
    PYTHIA_31M_MODEL,
    PYTHIA_70M_MODEL,
    PYTHIA_160M_MODEL,
    PYTHIA_410M_MODEL,
    PYTHIA_1_4B_MODEL,
    PYTHIA_2_8B_MODEL,
    PYTHIA_6_9B_MODEL,
    PYTHIA_12B_MODEL,
    QWEN_1_5_500M_CHAT_MODEL,
    QWEN_1_5_1_8B_CHAT_MODEL,
    QWEN_1_5_4B_CHAT_MODEL,
    QWEN_1_5_7B_CHAT_MODEL,
    QWEN_1_5_14B_CHAT_MODEL,
    QWEN_1_5_72B_CHAT_MODEL, CLAUDE_2_1_MODEL, CLAUDE_3_HAIKU_MODEL, CLAUDE_3_SONNET_MODEL, CLAUDE_3_OPUS_MODEL,
)
from src.llms.claude_model import Claude
from src.llms.gemini_pro_chat import GeminiProChat
from src.llms.gemini_pro_text import GeminiProText
from src.llms.gpt import GPT
from src.llms.hf_chat_model import HuggingFaceChatModel
from src.llms.hf_text_model import HuggingFaceTextModel
from src.llms.palm_two_chat import PaLMTwoChat
from src.llms.palm_two_text import PaLMTwoText
from src.models.types import LLMs, Tasks, Prompting
from src.prompting.chain_of_thought import ChainOfThought
from src.prompting.few_shot import FewShot
from src.prompting.null_shot import NullShot
from src.prompting.null_shot_after import NullShotAfter
from src.prompting.null_shot_chain_of_thought import NullShotChainOfThought
from src.prompting.null_shot_v1 import NullShotV1
from src.prompting.null_shot_v2 import NullShotV2
from src.prompting.null_shot_v3 import NullShotV3
from src.prompting.zero_shot import ZeroShot
from src.prompting.zero_shot_chain_of_thought import ZeroShotChainOfThought
from src.tasks.anli import ANLI
from src.tasks.aqua import AQuA
from src.tasks.commonsense_qa import CommonsenseQA
from src.tasks.gsm8k import GSM8K
from src.tasks.halueval_dialogue import HaluEvalDialogue
from src.tasks.halueval_general import HaluEvalGeneral
from src.tasks.halueval_qa import HaluEvalQA
from src.tasks.halueval_summarization import HaluEvalSummarization
from src.tasks.open_book_qa import OpenBookQA
from src.tasks.race_h import RACEHigh
from src.tasks.race_m import RACEMiddle
from src.tasks.strategyqa import StrategyQA
from src.tasks.svamp import SVAMP
from src.tasks.triviaqa import TriviaQA
from src.tasks.winogrande import Winogrande
from src.tasks.wmt_news_en_ja import WMTENJA
from src.tasks.wmt_news_ja_en import WMTJAEN


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
        case Prompting.null_shot_after:
            return NullShotAfter
        case Prompting.null_shot_v1:
            return NullShotV1
        case Prompting.null_shot_v2:
            return NullShotV2
        case Prompting.null_shot_v3:
            return NullShotV3
        case _:
            raise NotImplementedError(f"Prompting {prompting.value} not implemented")


def get_model(model: LLMs, client: Client | Pipeline | Anthropic = None):
    match model:
        case LLMs.gpt_three_point_five_turbo | LLMs.gpt_four_turbo:
            return GPT(client)
        case LLMs.palm_two_text:
            return PaLMTwoText()
        case LLMs.palm_two_chat:
            return PaLMTwoChat()
        case LLMs.gemini_pro_text:
            return GeminiProText()
        case LLMs.gemini_pro_chat:
            return GeminiProChat()
        case LLMs.claude_2_1 | LLMs.claude_3_haiku | LLMs.claude_3_sonnet | LLMs.claude_3_opus:
            return Claude(client)
        case (
        LLMs.llama_two_seven
        | LLMs.llama_two_thirteen
        | LLMs.llama_two_seventy
        | LLMs.pythia_14m
        | LLMs.pythia_31m
        | LLMs.pythia_70m
        | LLMs.pythia_160m
        | LLMs.pythia_410m
        | LLMs.pythia_1b
        | LLMs.pythia_1_4b
        | LLMs.pythia_2_8b
        | LLMs.pythia_6_9b
        | LLMs.pythia_12b
        ):
            return HuggingFaceTextModel(client)
        case (
        LLMs.llama_two_chat_seven
        | LLMs.llama_two_chat_thirteen
        | LLMs.llama_two_chat_seventy
        | LLMs.qwen_1_5_500m_chat
        | LLMs.qwen_1_5_1_8b_chat
        | LLMs.qwen_1_5_4b_chat
        | LLMs.qwen_1_5_7b_chat
        | LLMs.qwen_1_5_14b_chat
        | LLMs.qwen_1_5_72b_chat
        ):
            return HuggingFaceChatModel(client)
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

        case LLMs.gemini_pro_text:
            return GEMINI_PRO_TEXT_MODEL
        case LLMs.gemini_pro_chat:
            return GEMINI_PRO_CHAT_MODEL

        case LLMs.claude_2_1:
            return CLAUDE_2_1_MODEL
        case LLMs.claude_3_haiku:
            return CLAUDE_3_HAIKU_MODEL
        case LLMs.claude_3_sonnet:
            return CLAUDE_3_SONNET_MODEL
        case LLMs.claude_3_opus:
            return CLAUDE_3_OPUS_MODEL

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

        case LLMs.pythia_14m:
            return PYTHIA_14M_MODEL
        case LLMs.pythia_31m:
            return PYTHIA_31M_MODEL
        case LLMs.pythia_70m:
            return PYTHIA_70M_MODEL
        case LLMs.pythia_160m:
            return PYTHIA_160M_MODEL
        case LLMs.pythia_410m:
            return PYTHIA_410M_MODEL
        case LLMs.pythia_1b:
            return PYTHIA_1B_MODEL
        case LLMs.pythia_1_4b:
            return PYTHIA_1_4B_MODEL
        case LLMs.pythia_2_8b:
            return PYTHIA_2_8B_MODEL
        case LLMs.pythia_6_9b:
            return PYTHIA_6_9B_MODEL
        case LLMs.pythia_12b:
            return PYTHIA_12B_MODEL

        case LLMs.qwen_1_5_500m_chat:
            return QWEN_1_5_500M_CHAT_MODEL
        case LLMs.qwen_1_5_1_8b_chat:
            return QWEN_1_5_1_8B_CHAT_MODEL
        case LLMs.qwen_1_5_4b_chat:
            return QWEN_1_5_4B_CHAT_MODEL
        case LLMs.qwen_1_5_7b_chat:
            return QWEN_1_5_7B_CHAT_MODEL
        case LLMs.qwen_1_5_14b_chat:
            return QWEN_1_5_14B_CHAT_MODEL
        case LLMs.qwen_1_5_72b_chat:
            return QWEN_1_5_72B_CHAT_MODEL

        case _:
            raise NotImplementedError(f"Model {model.value} not implemented")


def get_task(task: Tasks):
    match task:
        case Tasks.TriviaQA:
            return TriviaQA
        case Tasks.ANLI:
            return ANLI
        case Tasks.WMTJAEN:
            return WMTJAEN
        case Tasks.WMTENJA:
            return WMTENJA
        case Tasks.RACE_H:
            return RACEHigh
        case Tasks.RACE_M:
            return RACEMiddle
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
        case Tasks.HaluEvalGeneral:
            return HaluEvalGeneral
        case Tasks.HaluEvalDialogue:
            return HaluEvalDialogue
        case Tasks.HaluEvalQA:
            return HaluEvalQA
        case Tasks.HaluEvalSummarization:
            return HaluEvalSummarization
        case _:
            raise NotImplementedError(f"Task {task.value} not implemented")
