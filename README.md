# Null-Shot Prompting: Rethinking Prompting Large Language Models With Hallucination

This repository contains the code for the paper "Null-Shot Prompting: Rethinking Prompting Large Language Models With Hallucination" accepted at [EMNLP 2024 (Main, long)](https://2024.emnlp.org/program/accepted_main_conference/).

For data and analysis code, please refer to the [data repository](https://github.com/Pittawat2542/null-shot-results).

## Authors
Pittawat Taveekitworachai, Febri Abdullah, and Ruck Thawonmas

## Abstract

This paper investigates an interesting phenomenon where we observe performance increases in large language models (LLMs) when providing a prompt that causes and exploits hallucination. We propose null-shot prompting, a counter-intuitive approach where we deliberately instruct LLMs to reference a null, non-existent, section. We evaluate null-shot prompting across a variety of tasks, including arithmetic reasoning, commonsense reasoning, and reading comprehension. Notably, we observe a substantial increase in performance in arithmetic reasoning tasks for various models, with up to a 44.62% increase compared to a baseline in one model. Additional experiments on more complex mathematical problem-solving and hallucination detection benchmarks also reveal similar benefits from this approach. Furthermore, we explore the effects of combining reasoning, which typically mitigates hallucination, with hallucination within the prompt and find several cases of performance improvements. We hope this paper stimulates further interest, investigation, and discussion on how hallucination in prompts may not only affect LLMs but, in certain cases, enhance their performance.

## File structure
```
.
├── .env.example # Example of the environment file
├── .gitignore # Git ignore file
├── datasets # Supported evaluation datasets
├── main.py # The main entry point of the program, new dataset and model should be added here first
├── requirements.txt # Required Python packages
└── src # Source code
    ├── __init__.py # Initialization file
    ├── config.py # Configuration file, new dataset and model should be added here first
    ├── evaluation_facade.py # Facade for evaluation, new dataset and model should be added here first
    ├── llms # Wrapper for LLMs
    ├── models # OOP models
    ├── prompting # Wrapper for prompt engineering approaches
    └── tasks # Wrapper for evaluation   tasks
```

## Installation and Usage
0. Create a virtual environment (if needed):
```bash
conda create -n null-shot python=3.11
```
and activate it:
```bash
conda activate null-shot
```
1. Copy `.env.example` and rename it to `.env`. Follow instructions respective model providers to get the necessary API keys.
2. Install the requirements:
```bash
pip install -r requirements.txt
```
3. Main program is `main.py`. It can be run with the following command:
```bash
python main.py <command> <options>
```
The main program supports the following commands:
- `--help`, `-h`: Show help message
- `--models`, `-m`: [Required] Models to perform tasks `[gpt-3.5-turbo|gpt-4-turbo|palm-2-text|palm-2-chat|gemini-pro-text|gemini-pro-chat|claude-2.1|claude-3-haiku|claude-3-sonnet|claude-3-opus|llama-2-7b|llama-2-13b|llama-2-70b|llama-2-chat-7b|llama-2-chat-13b|llama-2-chat-70b|pythia-14m|pythia-31m|pythia-70m|pythia-160m|pythia-410m|pythia-1b|pythia-1.4b|pythia-2.8b|pythia-6.9b|pythia-12b|qwen-1.5-0.5b-chat|qwen-1.5-1.8b-chat|qwen-1.5-4b-chat|qwen-1.5-7b-chat|qwen-1.5-14b-chat|qwen-1.5-32b-chat|qwen-1.5-72b-chat|all]`
- `--tasks`, `-t`: [Required] Tasks to perform `[all|triviaqa|anli|wmt-ja-en|wmt-en-ja|race-h|race-m|winogrande|csqa|strategyqa|openbookqa|aqua|gsm8k|svamp|halueval-general|halueval-dialogue|halueval-qa|halueval-summarization|math-algebra|math-count-prob|math-geometry|math-number|math-int-algebra|math-pre-algebra|math-pre-calc]`
- `--prompting`, `-p`: [Required] Prompting technique to use `[zero-shot|few-shot|null-shot|cot|zero-shot-cot|null-shot-cot|null-shot-after|null-shot-v1|null-shot-v2|null-shot-v3|all]`
- `--output-path`, `-r`: Path to store results `[default: results]`
- `--evaluate-only`, `-e`: Evaluate only, do not run inference
- `--existing-result-root-path`, `-e-r`: Root path to existing results `[default:
                                  results]`

## Example
```bash
python main.py -m gpt-3.5-turbo -t math-algebra -p null-shot
```
