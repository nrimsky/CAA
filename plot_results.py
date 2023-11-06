"""
Plot results from experiments.

Usage:
python plot_results.py --type in_distribution --few_shot none --layers 15 16 17 --multipliers -1 0 1 --max_new_tokens 100 --model_size 7b --n_test_datapoints 1000 --add_every_token_position
"""

import json
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import matplotlib.cm as cm
import numpy as np
import argparse
from utils.helpers import SteeringSettings

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))


def get_data(
    layer: int,
    multiplier: int,
    settings: SteeringSettings,
) -> Dict[str, Any]:
    if settings.type != "out_of_distribution":
        directory = os.path.join(WORKING_DIR, "results")
    else:
        directory = os.path.join(WORKING_DIR, "analysis", "scored_results")
    filenames = settings.filter_result_files_by_suffix(
        directory, layer=layer, multiplier=multiplier
    )
    if len(filenames) > 1:
        print(f"[WARN] >1 filename found for filter {settings}")
    if len(filenames) == 0:
        print(f"[WARN] no filenames found for filter {settings}")
        return []
    with open(filenames[0], "r") as f:
        return json.load(f)


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
    layer: int, multipliers: List[float], settings: SteeringSettings
):
    few_shot_options = [
        ("positive", "Sycophantic few-shot prompt"),
        ("negative", "Non-sycophantic few-shot prompt"),
        ("none", "No few-shot prompt"),
    ]
    settings.few_shot = None
    save_to = os.path.join(
        WORKING_DIR,
        "analysis",
        f"{settings.make_result_save_suffix(layer=layer)}.svg",
    )
    plt.clf()
    plt.figure(figsize=(6, 6))
    all_results = {}
    for few_shot, label in few_shot_options:
        settings.few_shot = few_shot
        try:
            res_list = []
            for multiplier in multipliers:
                results = get_data(layer, multiplier, settings)
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
            all_results[few_shot] = res_list
        except:
            print(f"[WARN] Missing data for few_shot={few_shot} for layer={layer}")
    plt.legend()
    plt.xlabel("Multiplier")
    plt.ylabel("Probability of sycophantic answer to A/B question")
    plt.tight_layout()
    plt.savefig(save_to, format="svg")
    plt.savefig(save_to.replace("svg", "png"), format="png")
    # Save data in all_results used for plotting as .txt
    with open(save_to.replace(".svg", ".txt"), "w") as f:
        for few_shot, res_list in all_results.items():
            for multiplier, score in res_list:
                f.write(f"{few_shot}\t{multiplier}\t{score}\n")


def plot_truthful_qa_data_for_layer(
    layer: int, multipliers: List[float], settings: SteeringSettings
):
    save_to = os.path.join(
        WORKING_DIR,
        "analysis",
        f"{settings.make_result_save_suffix(layer=layer)}.svg",
    )
    res_per_category = defaultdict(list)
    for multiplier in multipliers:
        results = get_data(layer, multiplier, settings)
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
    plt.tight_layout()
    plt.savefig(save_to, format="svg")
    plt.savefig(save_to.replace("svg", "png"), format="png")
    # Save data in res_per_category used for plotting as .txt
    with open(save_to.replace(".svg", ".txt"), "w") as f:
        for category, res_list in res_per_category.items():
            for multiplier, score in res_list:
                f.write(f"{category}\t{multiplier}\t{score}\n")


def plot_out_of_distribution_data(
    layer: int, multipliers: List[float], settings: SteeringSettings
):
    save_to = os.path.join(
        WORKING_DIR,
        "analysis",
        f"{settings.make_result_save_suffix(layer=layer)}.svg",
    )
    plt.clf()
    plt.figure(figsize=(6, 6))
    res_list = []
    for multiplier in multipliers:
        results = get_data(layer, multiplier, settings)
        if len(results) == 0:
            continue
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
    plt.tight_layout()
    plt.savefig(save_to, format="svg")
    plt.savefig(save_to.replace("svg", "png"), format="png")
    # Save data in res_list used for plotting as .txt
    with open(save_to.replace(".svg", ".txt"), "w") as f:
        for multiplier, score in res_list:
            f.write(f"{multiplier}\t{score}\n")


def plot_per_layer_data_in_distribution(
    layers: List[int], multipliers: List[float], settings: SteeringSettings
):
    plt.clf()
    plt.figure(figsize=(6, 6))
    save_to = os.path.join(
        WORKING_DIR,
        "analysis",
        f"{settings.make_result_save_suffix()}.svg",
    )
    # Get a colormap and generate a sequence of colors from that colormap
    colors = cm.rainbow(np.linspace(0, 1, len(layers)))

    full_res = {}
    for index, layer in enumerate(layers):
        res_list = []
        for multiplier in multipliers:
            results = get_data(layer, multiplier, settings)
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
        full_res[layer] = res_list
    plt.legend()
    plt.xlabel("Multiplier")
    plt.ylabel("Probability of sycophantic answer to A/B question")
    plt.tight_layout()
    plt.savefig(save_to, format="svg")
    plt.savefig(save_to.replace("svg", "png"), format="png")
    # Save data in res_list used for plotting as .txt
    with open(save_to.replace(".svg", ".txt"), "w") as f:
        for layer, multiplier_scores in full_res.items():
            for multiplier, score in multiplier_scores:
                f.write(f"{layer}\t{multiplier}\t{score}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=["in_distribution", "out_of_distribution", "truthful_qa"],
    )
    parser.add_argument(
        "--few_shot",
        type=str,
        choices=["positive", "negative", "unbiased", "none"],
        default="none",
    )
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--multipliers", nargs="+", type=float, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--do_projection", action="store_true", default=False)
    parser.add_argument("--override_vector", type=int, default=None)
    parser.add_argument("--override_vector_model", type=str, default=None)
    parser.add_argument("--use_base_model", action="store_true", default=False)
    parser.add_argument("--model_size", type=str, choices=["7b", "13b"], default="7b")
    parser.add_argument("--n_test_datapoints", type=int, default=None)
    parser.add_argument("--add_every_token_position", action="store_true", default=False)
    parser.add_argument("--override_model_weights_path", type=str, default=None)


    args = parser.parse_args()

    steering_settings = SteeringSettings()
    steering_settings.type = args.type
    steering_settings.few_shot = args.few_shot
    steering_settings.max_new_tokens = args.max_new_tokens
    steering_settings.do_projection = args.do_projection
    steering_settings.override_vector = args.override_vector
    steering_settings.override_vector_model = args.override_vector_model
    steering_settings.use_base_model = args.use_base_model
    steering_settings.model_size = args.model_size
    steering_settings.n_test_datapoints = args.n_test_datapoints
    steering_settings.add_every_token_position = args.add_every_token_position
    steering_settings.override_model_weights_path = args.override_model_weights_path

    if steering_settings.type == "in_distribution":
        if len(args.layers) == 1:
            layer = args.layers[0]
            plot_in_distribution_data_for_layer(
                layer, args.multipliers, steering_settings
            )
        else:
            plot_per_layer_data_in_distribution(
                args.layers, args.multipliers, steering_settings
            )
    elif steering_settings.type == "out_of_distribution":
        for layer in args.layers:
            plot_out_of_distribution_data(layer, args.multipliers, steering_settings)
    elif steering_settings.type == "truthful_qa":
        for layer in args.layers:
            plot_truthful_qa_data_for_layer(layer, args.multipliers, steering_settings)
