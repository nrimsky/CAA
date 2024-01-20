"""
Example usage: python test_finetuned.py --test_type=out_of_distribution --save_filename="results_neg_finetune.json" --model_path="finetuned_models/finetuned_negative.pt"
"""

import json
import torch as t
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from dotenv import load_dotenv
import argparse
from typing import Dict, Literal, Optional
from tqdm import tqdm
from behaviors import (
    ALL_BEHAVIORS,
    get_ab_test_data,
    get_open_ended_test_data,
    get_results_dir,
    get_truthful_qa_data,
    get_finetuned_model_path,
    get_finetuned_model_results_path,
)
from utils.helpers import get_a_b_probs
from utils.tokenize import tokenize_llama_chat, E_INST

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")
MODEL = "meta-llama/Llama-2-7b-chat-hf"
DEVICE = "cuda:0" if t.cuda.is_available() else "cpu"


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 50) -> str:
    tokens = tokenize_llama_chat(tokenizer, prompt)
    tokens = t.tensor(tokens).unsqueeze(0).to(DEVICE)
    generated = model.generate(inputs=tokens, max_new_tokens=max_new_tokens, top_k=1)
    return tokenizer.batch_decode(generated)[0]


def get_logits(model, tokenizer, prompt: str, model_output: str) -> t.Tensor:
    tokens = tokenize_llama_chat(
        tokenizer,
        user_input=prompt,
        model_output=model_output,
    )
    tokens = t.tensor(tokens).unsqueeze(0).to(DEVICE)
    return model(tokens).logits


def process_item_ab(
    item: Dict[str, str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    a_token_id: int,
    b_token_id: int,
) -> Dict[str, str]:
    prompt = item["question"]
    answer_matching_behavior = item["answer_matching_behavior"]
    answer_not_matching_behavior = item["answer_not_matching_behavior"]
    model_output = get_logits(model, tokenizer, prompt, "(")
    a_prob, b_prob = get_a_b_probs(model_output, a_token_id, b_token_id)
    return {
        "question": prompt,
        "answer_matching_behavior": answer_matching_behavior,
        "answer_not_matching_behavior": answer_not_matching_behavior,
        "a_prob": a_prob,
        "b_prob": b_prob,
    }


def process_item_open_ended(
    item: Dict[str, str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    a_token_id: int,
    b_token_id: int,
) -> Dict[str, str]:
    question = item["question"]
    model_output = generate(model, tokenizer, question, 100)
    return {
        "question": question,
        "model_output": model_output.split(E_INST)[-1].strip(),
        "raw_model_output": model_output,
    }


def process_item_truthful_qa(
    item: Dict[str, str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    a_token_id: int,
    b_token_id: int,
) -> Dict[str, str]:
    prompt = item["prompt"]
    correct = item["correct"]
    incorrect = item["incorrect"]
    category = item["category"]
    model_output = get_logits(model, tokenizer, prompt, "(")
    a_prob, b_prob = get_a_b_probs(model_output, a_token_id, b_token_id)
    return {
        "question": prompt,
        "correct": correct,
        "incorrect": incorrect,
        "a_prob": a_prob,
        "b_prob": b_prob,
        "category": category,
    }


def test_finetune(
    test_type: str,
    behavior: str,
    pos_or_neg: Optional[Literal["pos", "neg"]],
    layer=None,
):
    save_results_dir = get_results_dir(behavior)
    finetuned_model_path = get_finetuned_model_path(behavior, pos_or_neg, layer)
    save_filename = get_finetuned_model_results_path(behavior, pos_or_neg, layer)
    if not os.path.exists(save_results_dir):
        os.makedirs(save_results_dir)
    process_methods = {
        "ab": process_item_ab,
        "open_ended": process_item_open_ended,
        "truthful_qa": process_item_truthful_qa,
    }
    test_datasets = {
        "ab": get_ab_test_data(behavior),
        "open_ended": get_open_ended_test_data(behavior),
        "truthful_qa": get_truthful_qa_data(),
    }
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_auth_token=HUGGINGFACE_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, use_auth_token=HUGGINGFACE_TOKEN
    )
    model.load_state_dict(t.load(finetuned_model_path))
    model = model.to(DEVICE)
    a_token_id = tokenizer.convert_tokens_to_ids("A")
    b_token_id = tokenizer.convert_tokens_to_ids("B")
    test_data = test_datasets[test_type]
    results = []
    for item in tqdm(test_data):
        result = process_methods[test_type](
            item,
            model,
            tokenizer,
            a_token_id,
            b_token_id,
        )
        results.append(result)
    with open(
        save_filename,
        "w",
    ) as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=["ab", "open_ended", "truthful_qa"],
    )
    parser.add_argument(
        "--behavior",
        type=str,
        required=True,
        choices=ALL_BEHAVIORS,
    )
    parser.add_argument("--direction", type=str, choices=["pos", "neg"], required=False)
    args = parser.parse_args()
    test_finetune(
        args.type,
        args.behavior,
        args.direction,
    )