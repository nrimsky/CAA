"""
Data extracted from https://github.com/wusche1/CAA_hallucination/
"""

import json
import pandas as pd
import random

filenames = [
    "alluding_questioning.csv",
    "direct_questions.csv",
    "questioning_assuming_statement.csv",
]


def make_dataset(filename, add_accept_truth_questions=False):
    """
    `add_accept_truth_questions` is used to add questions that contrast accepting vs. rejecting true questions to correct for "just reject everything" bias
    """
    data = []
    df = pd.read_csv(filename)
    for _, row in df.iterrows():
        truth_question = row["truth_question"]
        fiction_question = row["fiction_question"]
        truth_acceptance = row["truth_accepetance"]
        truth_rejection = row["truth_rejection"]
        fiction_acceptance = row["fiction_accepetance"]
        fiction_rejection = row["fiction_rejection"]
        a_truth, b_truth = (
            (truth_acceptance, truth_rejection)
            if random.random() < 0.5
            else (truth_rejection, truth_acceptance)
        )
        a_fiction, b_fiction = (
            (fiction_acceptance, fiction_rejection)
            if random.random() < 0.5
            else (fiction_rejection, fiction_acceptance)
        )
        # matching_behavior = choosing fiction_acceptance for fiction_question + truth_rejection for truth_question
        # not_matching_behavior = choosing fiction_rejection for fiction_question + truth_acceptance for truth_question
        data.append(
            {
                "question": f"{fiction_question}\n\nChoices:\n (A) {a_fiction}\n (B) {b_fiction}",
                "answer_matching_behavior": "(A)"
                if a_fiction == fiction_acceptance
                else "(B)",
                "answer_not_matching_behavior": "(A)"
                if a_fiction == fiction_rejection
                else "(B)",
            }
        )
        if add_accept_truth_questions:
            data.append(
                {
                    "question": f"{truth_question}\n\nChoices:\n (A) {a_truth}\n (B) {b_truth}",
                    "answer_matching_behavior": "(A)"
                    if a_truth == truth_rejection
                    else "(B)",
                    "answer_not_matching_behavior": "(A)"
                    if a_truth == truth_acceptance
                    else "(B)",
                }
            )
    return data


def make_datasets():
    all_data = []
    for f in filenames:
        data = make_dataset(f)
        all_data.extend([{"type": f.split(".")[0], **d} for d in data])
    with open("dataset.json", "w") as f:
        json.dump(all_data, f, indent=4)


if __name__ == "__main__":
    make_datasets()
