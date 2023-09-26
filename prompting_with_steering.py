"""
Test activation steering on different datasets.

Usage:
python prompting_with_steering.py --type in_distribution --layers 15 20 25 --multipliers -10 -5 0 5 10 --max_new_tokens 100 
"""

import json
import torch as t
from llama_wrapper import Llama7BChatWrapper
from llm_generated_data.sycophancy_validation_prompts import PROMPTS
import os
from dotenv import load_dotenv
import argparse
from typing import List, Dict
from tqdm import tqdm

load_dotenv()

TEST_DATA_PATH = os.path.join("preprocessed_data", "test_dataset.json")
TRUTHFUL_QA_DATA_PATH = os.path.join("preprocessed_data", "truthful_qa_dataset.json")
HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")
SYSTEM_PROMPT = "You are a helpful, honest and concise assistant."
VECTORS_PATH = "vectors"
SAVE_RESULTS_PATH = "results"


def get_test_data():
    with open(TEST_DATA_PATH, "r") as f:
        data = json.load(f)
    return data


def get_truthful_qa_data():
    with open(TRUTHFUL_QA_DATA_PATH, "r") as f:
        data = json.load(f)
    return data


def get_steering_vector(layer):
    return t.load(os.path.join(VECTORS_PATH, f"vec_layer_{layer}.pt"))


def process_item_in_distribution(
    item: Dict[str, str], model: Llama7BChatWrapper, max_new_tokens: int
) -> Dict[str, str]:
    prompt = item["question"]
    answer_matching_behavior = item["answer_matching_behavior"]
    answer_not_matching_behavior = item["answer_not_matching_behavior"]
    model_output = model.generate_text(prompt, max_new_tokens=max_new_tokens)
    return {
        "question": prompt,
        "answer_matching_behavior": answer_matching_behavior,
        "answer_not_matching_behavior": answer_not_matching_behavior,
        "model_output": model_output.split("[/INST]")[-1].strip(),
    }


def process_item_out_of_distribution(
    prompt: str, model: Llama7BChatWrapper, max_new_tokens: int
) -> Dict[str, str]:
    model_output = model.generate_text(prompt, max_new_tokens=max_new_tokens)
    return {
        "question": prompt,
        "model_output": model_output.split("[/INST]")[-1].strip(),
    }


def process_item_truthful_qa(
    item: Dict[str, str], model: Llama7BChatWrapper, max_new_tokens: int
) -> Dict[str, str]:
    prompt = item["prompt"]
    correct = item["correct"]
    incorrect = item["incorrect"]
    correct_full_text = item["correct_full_text"]
    incorrect_full_text = item["incorrect_full_text"]
    model_output = model.generate_text(prompt, max_new_tokens=max_new_tokens)
    return {
        "question": prompt,
        "correct": correct,
        "incorrect": incorrect,
        "correct_full_text": correct_full_text,
        "incorrect_full_text": incorrect_full_text,
        "model_output": model_output.split("[/INST]")[-1].strip(),
    }


def test_steering(
    layers: List[int],
    multipliers: List[int],
    max_new_tokens: int,
    type: str,
    do_projection: bool = False,
):
    """
    layers: List of layers to test steering on.
    multipliers: List of multipliers to test steering with.
    max_new_tokens: Maximum number of tokens to generate.
    type: Type of test to run. One of "in_distribution", "out_of_distribution", "truthful_qa".
    do_projection: Whether to project activations onto orthogonal complement of steering vector.
    """
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
    model = Llama7BChatWrapper(HUGGINGFACE_TOKEN, SYSTEM_PROMPT)
    model.set_save_internal_decodings(False)
    test_data = get_data_methods[type]()
    for layer in layers:
        vector = get_steering_vector(layer)
        vector = vector.to(model.device)
        for multiplier in multipliers:
            results = []
            for item in tqdm(test_data, desc=f"Layer {layer}, multiplier {multiplier}"):
                model.reset_all()
                model.set_add_activations(
                    layer, multiplier * vector, do_projection=do_projection
                )
                result = process_methods[type](item, model, max_new_tokens)
                results.append(result)
            with open(
                os.path.join(
                    SAVE_RESULTS_PATH,
                    f"results_{type}_layer_{layer}_multiplier_{multiplier}.json",
                ),
                "w",
            ) as f:
                json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=["in_distribution", "out_of_distribution", "truthful_qa"],
    )
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--multipliers", nargs="+", type=int, required=True)
    parser.add_argument("--max_new_tokens", type=int, required=True)
    parser.add_argument("--do_projection", action="store_true", default=False)
    args = parser.parse_args()
    test_steering(
        args.layers,
        args.multipliers,
        args.max_new_tokens,
        args.type,
        args.do_projection,
    )
