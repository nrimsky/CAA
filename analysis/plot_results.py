"""
Plot results from experiments.

Usage:
python plot_results.py --multipliers -1.5 -1.0 -0.5 0 0.5 1.0 1.5 --type in_distribution --layers 15 20 25 --title "Effect of steering on A/B question answering"
"""

import json
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import matplotlib.cm as cm
import numpy as np
import argparse

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
    tot = 0
    for result in results:
        try:
            score_sum += float(result["score"])
            tot += 1
        except:
            print(f"[WARN] Skipping invalid score: {result['score']}")
    return score_sum / tot


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
    layer: int,
    multipliers: List[float],
    save_to: str,
    title: str = "Effect of steering on A/B question answering",
):
    few_shot_options = [
        ("positive", "Sycophantic few-shot prompt"),
        ("negative", "Non-sycophantic few-shot prompt"),
        ("none", "No few-shot prompt"),
    ]
    plt.clf()
    plt.figure(figsize=(6, 6))
    for few_shot, label in few_shot_options:
        try:
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
        except:
            print(f"[WARN] Missing data for few_shot={few_shot} for layer={layer}")
    plt.legend()
    plt.xlabel("Multiplier")
    plt.ylabel("Probability of sycophantic answer to A/B question")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_to)


def plot_truthful_qa_data_for_layer(
    layer: int,
    multipliers: List[float],
    save_to: str,
    few_shot: str = "none",
    title: str = "Effect of steering on TruthfulQA",
):
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
    colors = cm.rainbow(np.linspace(0, 1, len(res_per_category)))
    for idx, (category, res_list) in enumerate(res_per_category.items()):
        res_per_category[category].sort(key=lambda x: x[0])
        plt.plot(
            [x[0] for x in res_list],
            [x[1] for x in res_list],
            label=category,
            marker="o",
            linestyle="dashed",
            markersize=5,
            linewidth=2.5,
            color=colors[idx],
        )
    plt.legend()
    plt.xlabel("Multiplier")
    plt.ylabel("Probability of correct answer to A/B question")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_to)


def plot_out_of_distribution_data(
    layer: int,
    multipliers: List[float],
    save_to: str,
    few_shot: str = "none",
    title: str = "Effect of steering on Claude score",
):
    plt.clf()
    plt.figure(figsize=(6, 6))
    res_list = []
    for multiplier in multipliers:
        results = get_data(layer, "out_of_distribution", multiplier, few_shot)
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
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_to)


def plot_per_layer_data_in_distribution(
    layers: List[int],
    multipliers: List[float],
    save_to: str,
    few_shot: str = "none",
    title: str = "Effect of steering at different layers",
):
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
            color=colors[index],  # Set color for each line
        )
    plt.legend()
    plt.xlabel("Multiplier")
    plt.ylabel("Probability of sycophantic answer to A/B question")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_to)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--multipliers", nargs="+", type=float, required=True)
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=["in_distribution", "out_of_distribution", "truthful_qa"],
    )
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--title", type=str, required=True)
    parser.add_argument(
        "--few_shot",
        type=str,
        required=False,
        default="none",
        choices=["positive", "negative", "unbiased", "none"],
    )
    args = parser.parse_args()

    experiment_type = args.type
    title = args.title

    if experiment_type == "in_distribution":
        if len(args.layers) == 1:
            layer = args.layers[0]
            plot_in_distribution_data_for_layer(
                layer,
                args.multipliers,
                os.path.join(
                    PARENT_DIR, "analysis", f"in_distribution_layer_{layer}.png"
                ),
                f"{title}, layer {layer}",
            )
        else:
            plot_per_layer_data_in_distribution(
                args.layers,
                args.multipliers,
                os.path.join(PARENT_DIR, "analysis", f"in_distribution_all_layers.png"),
                args.few_shot,
                title,
            )
    elif experiment_type == "out_of_distribution":
        for layer in args.layers:
            plot_out_of_distribution_data(
                layer,
                args.multipliers,
                os.path.join(
                    PARENT_DIR, "analysis", f"out_of_distribution_layer_{layer}.png"
                ),
                args.few_shot,
                f"{title}, layer {layer}",
            )
    elif experiment_type == "truthful_qa":
        for layer in args.layers:
            plot_truthful_qa_data_for_layer(
                layer,
                args.multipliers,
                os.path.join(PARENT_DIR, "analysis", f"truthful_qa_layer_{layer}.png"),
                args.few_shot,
                f"{title}, layer {layer}",
            )
