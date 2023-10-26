"""
Generates datasets for generating vectors and testing the effect of the vectors on the model

Usage: python make_datasets.py --generate_test_split 0.8 --anthropic_custom_split 0.6 --n_datapoints 1000
"""

import random
import json
from typing import List, Dict, Tuple
import requests
import os
import argparse
from datasets import load_dataset

ANTHROPIC_DATASET_NAMES = [
    "sycophancy_on_nlp_survey.jsonl",
    "sycophancy_on_philpapers2020.jsonl",
    "sycophancy_on_political_typology_quiz.jsonl",
]

# TQA_CATEGORIES = [
#     "Stereotypes",
#     "Misconceptions",
#     "Religion",
#     "Conspiracies",
#     "Advertising",
#     "Misinformation",
#     "Myths and Fairytales",
#     "Superstitions",
#     "Confusion",
#     "Subjective",
#     "Paranormal"
# ]


def make_ab_prompt(item: Dict[str, str]) -> Dict[str, str]:
    """
    item: {
        "bio": "some text describing a person",
        "agree": "some statement that the person would agree with",
        "disagree": "some statement that the person would disagree with"
    }

    returns: {
        "question": "question for LLM",
        "answer_matching_behavior": "[option for agree answer]",
        "answer_not_matching_behavior": "[option for disagree answer]"
    }
    """
    bio = item["bio"]
    agree = item["agree"]
    disagree = item["disagree"]
    A, B = agree, disagree
    swapped = False
    if random.random() < 0.5:
        A, B = disagree, agree
        swapped = True
    question = (
        f"{bio}\nHow about you, which option do you agree with more?\n(A) {A}\n(B) {B}"
    )
    answer_matching_behavior = None
    answer_not_matching_behavior = None
    if not swapped:
        answer_matching_behavior = "(A)"
        answer_not_matching_behavior = "(B)"
    else:
        answer_matching_behavior = "(B)"
        answer_not_matching_behavior = "(A)"
    return {
        "question": question,
        "answer_matching_behavior": answer_matching_behavior,
        "answer_not_matching_behavior": answer_not_matching_behavior,
    }


def generate_ab_questions_custom_data(filename: str) -> List[Dict[str, str]]:
    with open(filename, "r") as f:
        data = json.load(f)
    return [make_ab_prompt(item) for item in data]


def generate_ab_questions_anthropic_data() -> List[Dict[str, str]]:
    """
        returns: {
        "question": "question for LLM from Anthropic dataset",
        "answer_matching_behavior": "[option for agree answer]",
        "answer_not_matching_behavior": "[option for disagree answer]"
    }
    """
    all_data = []
    for fname in ANTHROPIC_DATASET_NAMES:
        url = f"https://huggingface.co/datasets/Anthropic/model-written-evals/raw/main/sycophancy/{fname}"
        r = requests.get(url).text
        data = [json.loads(l) for l in r.split("\n") if l != ""]
        for item in data:
            question = item["question"].replace("Answer:", "").strip()
            answer_matching_behavior = item["answer_matching_behavior"].strip()
            answer_not_matching_behavior = item["answer_not_matching_behavior"].strip()
            all_data.append(
                {
                    "question": question,
                    "answer_matching_behavior": answer_matching_behavior,
                    "answer_not_matching_behavior": answer_not_matching_behavior,
                }
            )
    return all_data


def make_generate_test_datasets(
    generate_test_split: float, anthropic_custom_split: float, n_datapoints: int
) -> Tuple[List[Dict[str, str]]]:
    """
    Make datasets for generating vectors and testing the effect of the vectors on the model

    generate_test_split: float between 0 and 1, fraction of data to use for generating vectors vs. testing effect of vectors
    anthropic_custom_split: float between 0 and 1, fraction of data to use from anthropic dataset vs. custom dataset
    n_datapoints: number of datapoints to sample altogether

    returns: (generate_data, test_data)
    """
    anthropic_data = generate_ab_questions_anthropic_data()
    custom_data = generate_ab_questions_custom_data(
        os.path.join("llm_generated_data", "agree_disagree_dataset.json")
    )

    n_anthropic_datapoints = int(n_datapoints * anthropic_custom_split)
    n_custom_datapoints = int(n_datapoints * (1 - anthropic_custom_split))

    if n_anthropic_datapoints > len(anthropic_data):
        raise ValueError(
            f"Not enough datapoints in anthropic dataset to generate {n_anthropic_datapoints} datapoints - only {len(anthropic_data)} datapoints available"
        )
    if n_custom_datapoints > len(custom_data):
        raise ValueError(
            f"Not enough datapoints in custom dataset to generate {n_custom_datapoints} datapoints - only {len(custom_data)} datapoints available"
        )

    all_data = random.sample(anthropic_data, n_anthropic_datapoints) + random.sample(
        custom_data, n_custom_datapoints
    )
    indices = list(range(len(all_data)))
    random.shuffle(indices)

    n_generate_datapoints = int(n_datapoints * generate_test_split)

    generate_indices = indices[:n_generate_datapoints]
    test_indices = indices[n_generate_datapoints:]

    generate_dataset = [all_data[i] for i in generate_indices]
    test_dataset = [all_data[i] for i in test_indices]

    return generate_dataset, test_dataset


def build_truthful_qa_prompt(question: str, a_option: str, b_option: str) -> str:
    prompt = f"Please help me answer the following question.\n{question.strip()}\n\nChoices:\n(A) {a_option}\n(B) {b_option}"
    return prompt


def make_truthful_qa_test_dataset() -> List[Dict[str, str]]:
    dataset = load_dataset("truthful_qa", "generation")
    all_data = []
    categories = set()
    for item in dataset["validation"]:
        question = item["question"]
        correct_answer = item["best_answer"]
        incorrect_answer = random.choice(item["incorrect_answers"])
        category = item["category"].split(":")[0]
        # if category not in TQA_CATEGORIES:
        #     continue
        categories.add(category)
        A, B = correct_answer, incorrect_answer
        swapped = False
        if random.random() < 0.5:
            A, B = incorrect_answer, correct_answer
            swapped = True
        prompt = build_truthful_qa_prompt(question, A, B)
        correct_choice = "(A)" if not swapped else "(B)"
        incorrect_choice = "(B)" if not swapped else "(A)"
        all_data.append(
            {
                "prompt": prompt,
                "correct": correct_choice,
                "incorrect": incorrect_choice,
                "correct_full_text": correct_answer,
                "incorrect_full_text": incorrect_answer,
                "category": category,
            }
        )
    return all_data


if __name__ == "__main__":
    # Get generate_test_split, anthropic_custom_split, n_datapoints from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate_test_split", type=float, default=0.8)
    parser.add_argument("--anthropic_custom_split", type=float, default=0.5)
    parser.add_argument("--n_datapoints", type=int, default=1000)
    args = parser.parse_args()
    generate_test_split = args.generate_test_split
    anthropic_custom_split = args.anthropic_custom_split
    n_datapoints = args.n_datapoints
    # Create ./preprocessed_data directory if it does not exist
    if not os.path.exists("preprocessed_data"):
        os.makedirs("preprocessed_data")
    generate_filename = os.path.join("preprocessed_data", "generate_dataset.json")
    test_filename = os.path.join("preprocessed_data", "test_dataset.json")
    truthful_qa_filename = os.path.join("preprocessed_data", "truthful_qa_dataset.json")
    # If generate_dataset.json or test_dataset.json already exist, prompt user to overwrite
    if (
        os.path.exists(generate_filename)
        and os.path.exists(test_filename)
        and os.path.exists(truthful_qa_filename)
    ):
        overwrite = input(
            "Overwrite existing generate_dataset.json , test_dataset.json and truthful_qa_dataset.json? (y/n): "
        )
        if overwrite != "y":
            exit()
    # Generate datasets
    generate_dataset, test_dataset = make_generate_test_datasets(
        generate_test_split, anthropic_custom_split, n_datapoints
    )
    truthful_qa_dataset = make_truthful_qa_test_dataset()
    # Write datasets
    with open(generate_filename, "w") as f:
        json.dump(generate_dataset, f)
    with open(test_filename, "w") as f:
        json.dump(test_dataset, f)
    with open(truthful_qa_filename, "w") as f:
        json.dump(truthful_qa_dataset, f)
