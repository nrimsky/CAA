import json
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import matplotlib.cm as cm
import numpy as np

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_data(layer: int, type: str, multiplier: float, few_shot: str) -> Dict[str, Any]:
    if type != "out_of_distribution":
        fname = f"results_type_{type}_layer_{layer}_multiplier_{multiplier}_few_shot_{few_shot}.json"
        fname = os.path.join(PARENT_DIR, "results", fname)
    else:
        fname = f"results_type_{type}_layer_{layer}_multiplier_{multiplier}_few_shot_{few_shot}_scored.json"
        fname = os.path.join(PARENT_DIR, "analysis", "scored_results", fname)
    with open(fname, "r") as f:
        results = json.load(f)
    return results

def get_avg_score(results: Dict[str, Any]) -> float:
    score_sum = 0.0
    for result in results:
        try:
            score_sum += float(result["score"])
        except:
            score = input(f"Enter score.\nQuestion: {result['question']}\nAnswer: {result['model_output']}\n>>> ")
            score_sum += float(score)
    return score_sum / len(results)


def get_avg_key_prob(results: Dict[str, Any], key: str) -> float:
    match_key_prob_sum = 0.0
    for result in results:
        matching_value = result[key]
        denom = result["a_prob"] + result["b_prob"]
        if "A" in matching_value:
            match_key_prob_sum += result["a_prob"] / denom
        elif "B" in matching_value:
            match_key_prob_sum += result["b_prob"] / denom
    return match_key_prob_sum / len(results)


def plot_in_distribution_data_for_layer(
    layer: int, multipliers: List[float], save_to: str
):
    few_shot_options = [
        ("positive", "Sycophantic few-shot prompt"),
        ("negative", "Non-sycophantic few-shot prompt"),
        ("none", "No few-shot prompt")
    ]
    plt.clf()
    plt.figure(figsize=(6, 6))
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
            marker="o",
            linestyle="dashed",
            markersize=5,
            linewidth=2.5,
        )
    plt.legend()
    plt.xlabel("Multiplier")
    plt.ylabel("Probability of sycophantic answer to A/B question")
    plt.title(f"Effect of steering at layer {layer}")
    plt.tight_layout()
    plt.savefig(save_to)


def plot_truthful_qa_data_for_layer(layer: int, multipliers: List[float], few_shot: str, save_to: str):
    res_per_category = defaultdict(list)
    for multiplier in multipliers:
        results = get_data(layer, "truthful_qa", multiplier, few_shot)
        categories = set([item["category"] for item in results])
        for category in categories:
            category_results = [
                item for item in results if item["category"] == category
            ]
            avg_key_prob = get_avg_key_prob(category_results, "correct")
            res_per_category[category].append((multiplier, avg_key_prob))
    plt.clf()
    plt.figure(figsize=(6, 6))
    for category, res_list in res_per_category.items():
        res_per_category[category].sort(key=lambda x: x[0])
        plt.plot(
            [x[0] for x in res_list],
            [x[1] for x in res_list],
            label=category,
            marker="o",
            linestyle="dashed",
            markersize=5,
            linewidth=2.5,
        )
    plt.legend()
    plt.xlabel("Multiplier")
    plt.ylabel("Probability of correct answer to A/B question")
    plt.title(f"Effect of steering at layer {layer} on TruthfulQA")
    plt.tight_layout()
    plt.savefig(save_to)


def plot_out_of_distribution_data(
    layer: int, multipliers: List[float], save_to: str
):
    plt.clf()
    plt.figure(figsize=(6, 6))
    res_list = []
    for multiplier in multipliers:
        results = get_data(layer, "out_of_distribution", multiplier, "none")
        avg_score = get_avg_score(results)
        res_list.append((multiplier, avg_score))
    res_list.sort(key=lambda x: x[0])
    plt.plot(
        [x[0] for x in res_list],
        [x[1] for x in res_list],
        marker="o",
        linestyle="dashed",
        markersize=5,
        linewidth=2.5,
    )
    plt.xlabel("Multiplier")
    plt.ylabel("Claude score")
    plt.title("Effect of steering on Claude score")
    plt.tight_layout()
    plt.savefig(save_to)

def plot_per_layer_data_in_distribution(layers: List[int], multipliers: List[float], save_to: str, few_shot: str = "none"):
    plt.clf()
    plt.figure(figsize=(6, 6))
    
    # Get a colormap and generate a sequence of colors from that colormap
    colors = cm.rainbow(np.linspace(0, 1, len(layers)))
    
    for index, layer in enumerate(layers):
        res_list = []
        for multiplier in multipliers:
            results = get_data(layer, "in_distribution", multiplier, few_shot)
            avg_key_prob = get_avg_key_prob(results, "answer_matching_behavior")
            res_list.append((multiplier, avg_key_prob))
        res_list.sort(key=lambda x: x[0])
        plt.plot(
            [x[0] for x in res_list],
            [x[1] for x in res_list],
            label=f"Layer {layer}",
            marker="o",
            linestyle="dashed",
            markersize=5,
            linewidth=2.5,
            color=colors[index]  # Set color for each line
        )
    plt.legend()
    plt.xlabel("Multiplier")
    plt.ylabel("Probability of sycophantic answer to A/B question")
    plt.title(f"Effect of steering at different layers")
    plt.tight_layout()
    plt.savefig(save_to)

if __name__ == "__main__":
    multipliers = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    all_layers = list(range(14, 32))
    main_layer = 16

    plot_in_distribution_data_for_layer(
        main_layer,
        multipliers,
        os.path.join(PARENT_DIR, "analysis", f"in_distribution_layer_{main_layer}.png"),
    )

    plot_truthful_qa_data_for_layer(
        main_layer,
        multipliers,
        "none",
        os.path.join(PARENT_DIR, "analysis", f"truthful_qa_layer_{main_layer}.png"),
    )

    plot_out_of_distribution_data(
        main_layer,
        multipliers,
        os.path.join(PARENT_DIR, "analysis", f"out_of_distribution_layer_{main_layer}.png"),
    )

    plot_per_layer_data_in_distribution(
        all_layers,
        multipliers,
        os.path.join(PARENT_DIR, "analysis", f"in_distribution_all_layers.png")
    )