import json
import torch as t
from llm_generated_data.sycophancy_validation_prompts import PROMPTS
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from dotenv import load_dotenv
import argparse
from typing import Dict, Tuple, Optional, List
from tqdm import tqdm
from utils.helpers import get_a_b_probs
from utils.tokenize_llama import tokenize_llama

load_dotenv()

TEST_DATA_PATH = os.path.join("preprocessed_data", "test_dataset.json")
TRUTHFUL_QA_DATA_PATH = os.path.join("preprocessed_data", "truthful_qa_dataset.json")
HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")
SYSTEM_PROMPT = "You are a helpful, honest and concise assistant."
SAVE_RESULTS_PATH = "results_finetuned"
MODEL = "meta-llama/Llama-2-7b-chat-hf"
DEVICE = "cuda:0" if t.cuda.is_available() else "cpu"

def generate(model, tokenizer, prompt: str, max_new_tokens: int = 50) -> str:
    tokens = tokenize_llama(
        tokenizer,
        SYSTEM_PROMPT,
        [(prompt, None)],
        chat_model=True,
    )
    tokens = t.tensor(tokens).unsqueeze(0).to(DEVICE)
    generated = model.generate(
        inputs=tokens, max_new_tokens=max_new_tokens, top_k=1
    )
    return tokenizer.batch_decode(generated)[0]

def get_logits(model, tokenizer, history: List[Tuple[str, Optional[str]]]) -> t.Tensor:
    tokens = tokenize_llama(
        tokenizer,
        SYSTEM_PROMPT,
        history,
        no_final_eos=True,
        chat_model=True,
    )
    tokens = t.tensor(tokens).unsqueeze(0).to(DEVICE)
    return model(tokens).logits

def get_test_data():
    with open(TEST_DATA_PATH, "r") as f:
        data = json.load(f)
    return data


def get_truthful_qa_data():
    with open(TRUTHFUL_QA_DATA_PATH, "r") as f:
        data = json.load(f)
    return data

def process_item_in_distribution(
    item: Dict[str, str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int,
    a_token_id: int,
    b_token_id: int,
) -> Dict[str, str]:
    prompt = item["question"]
    answer_matching_behavior = item["answer_matching_behavior"]
    answer_not_matching_behavior = item["answer_not_matching_behavior"]
    model_output = get_logits(model, tokenizer, [(prompt, "My answer is (")])
    a_prob, b_prob = get_a_b_probs(model_output, a_token_id, b_token_id)
    return {
        "question": prompt,
        "answer_matching_behavior": answer_matching_behavior,
        "answer_not_matching_behavior": answer_not_matching_behavior,
        "a_prob": a_prob,
        "b_prob": b_prob,
    }


def process_item_out_of_distribution(
    prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int,
    a_token_id: int,
    b_token_id: int,
) -> Dict[str, str]:
    model_output = generate(model, tokenizer, prompt, max_new_tokens)
    return {
        "question": prompt,
        "model_output": model_output.split("[/INST]")[-1].strip(),
        "raw_model_output": model_output,
    }


def process_item_truthful_qa(
    item: Dict[str, str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int,
    a_token_id: int,
    b_token_id: int,
) -> Dict[str, str]:
    prompt = item["prompt"]
    correct = item["correct"]
    incorrect = item["incorrect"]
    category = item["category"]
    model_output = get_logits(model, tokenizer, [(prompt, "My answer is (")])
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
    test_type: str, max_new_tokens: int, save_filename: str
):
    if not os.path.exists(SAVE_RESULTS_PATH):
        os.makedirs(SAVE_RESULTS_PATH)
    process_methods = {
        "in_distribution": process_item_in_distribution,
        "out_of_distribution": process_item_out_of_distribution,
        "truthful_qa": process_item_truthful_qa,
    }
    get_data_methods = {
        "in_distribution": get_test_data,
        "out_of_distribution": lambda: PROMPTS,
        "truthful_qa": get_truthful_qa_data,
    }

    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_auth_token=HUGGINGFACE_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, use_auth_token=HUGGINGFACE_TOKEN)
    model_path = "finetuned_models/finetuned_0.pt"
    model.load_state_dict(t.load(model_path))
    model = model.to(DEVICE)
    a_token_id = model.tokenizer.convert_tokens_to_ids("A")
    b_token_id = model.tokenizer.convert_tokens_to_ids("B")
    model.set_save_internal_decodings(False)
    test_data = get_data_methods[test_type]()
    results = []
    for item in tqdm(test_data):
        result = process_methods[test_type](
            item,
            model,
            tokenizer,
            max_new_tokens,
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
        "--test_type",
        type=str,
        required=True,
        choices=["in_distribution", "out_of_distribution", "truthful_qa"],
    )
    parser.add_argument("--max_new_tokens", type=int, required=False, default=50)
    parser.add_argument("--save_filename", type=str, required=True)
    args = parser.parse_args()
    test_finetune(
        args.test_type,
        args.max_new_tokens,
        args.save_filename,
    )