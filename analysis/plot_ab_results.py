import json
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import os

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_data(layer: int, type: str, multiplier: float, few_shot: str) -> Dict[str, Any]:
    fname = f"results_type_{type}_layer_{layer}_multiplier_{multiplier}_few_shot_{few_shot}.json"
    fname = os.path.join(PARENT_DIR, "results", fname)
    with open(fname, "r") as f:
        results = json.load(f)
    return results


def get_avg_key_prob(results: Dict[str, Any], key: str) -> float:
    match_key_prob_sum = 0.0
    for result in results:
        matching_value = result[key]
        denom = result["a_prob"] + result["b_prob"]
        if "A" in matching_value:
            match_key_prob_sum += (result["a_prob"] / denom)
        elif "B" in matching_value:
            match_key_prob_sum += (result["b_prob"] / denom)
    return match_key_prob_sum / len(results)


def plot_in_distribution_data_for_layer(
    layer: int, multipliers: List[float], save_to: str
):
    few_shot_options = [
        ("positive", "Sycophantic few-shot prompt"),
        ("negative", "Non-sycophantic few-shot prompt"),
        ("unbiased", "Neutral few-shot prompt"),
    ]
    plt.clf()
    plt.figure(figsize=(5, 5))
    for few_shot, label in few_shot_options:
        res_list = []
        for multiplier in multipliers:
            results = get_data(layer, "in_distribution", multiplier, few_shot)
            avg_key_prob = get_avg_key_prob(results, "answer_matching_behavior")
            res_list.append((multiplier, avg_key_prob))
        res_list.sort(key=lambda x: x[0])
        plt.plot(
            [x[0] for x in res_list],
            [x[1] for x in res_list],
            label=label,
            marker="x",
            linestyle="dashed",
        )
    plt.legend()
    plt.xlabel("Multiplier")
    plt.ylabel("Probability of sycophantic answer to A/B question")
    plt.title(f"Effect of steering at layer {layer}")
    plt.tight_layout()
    plt.savefig(save_to)

def plot_truthful_qa_data_for_layer(
    layer: int, multipliers: List[float], save_to: str
):
    res_list = []
    for multiplier in multipliers:
        results = get_data(layer, "truthful_qa", multiplier, "unbiased")
        avg_key_prob = get_avg_key_prob(results, "correct")
        res_list.append((multiplier, avg_key_prob))
    res_list.sort(key=lambda x: x[0])
    plt.clf()
    plt.plot(
        [x[0] for x in res_list],
        [x[1] for x in res_list],
        marker="x",
        linestyle="dashed",
    )
    plt.xlabel("Multiplier")
    plt.ylabel("Probability of correct answer to A/B question")
    plt.title(f"Effect of steering at layer {layer} on TruthfulQA")
    plt.tight_layout()
    plt.savefig(save_to)

if __name__ == "__main__":
    multipliers = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    save_to = "steering_in_distribution_layer_15.png"
    plot_in_distribution_data_for_layer(15, multipliers, save_to)
    save_to = "steering_truthful_qa_layer_15.png"
    plot_truthful_qa_data_for_layer(15, multipliers, save_to)
