"""
Plot results from behavioral evaluations under steering.

Example usage:
python plot_results.py --layers 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 --multipliers -1 0 1 --behavior sycophancy --type ab
"""

import json
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import matplotlib.cm as cm
import numpy as np
import argparse
from steering_settings import SteeringSettings
from behaviors import get_results_dir, get_analysis_dir, ALL_BEHAVIORS


def get_data(
    layer: int,
    multiplier: int,
    settings: SteeringSettings,
) -> Dict[str, Any]:
    directory = get_results_dir(settings.behavior)
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


def plot_ab_results_for_layer(
    layer: int, multipliers: List[float], settings: SteeringSettings
):
    system_prompt_options = [
        ("pos", f"{settings.behavior} system prompt"),
        ("neg", f"{settings.behavior} system prompt"),
        (None, f"No system prompt"),
    ]
    settings.system_prompt = None
    save_to = os.path.join(
        get_analysis_dir(settings.behavior),
        f"{settings.make_result_save_suffix(layer=layer)}.svg",
    )
    plt.clf()
    plt.figure(figsize=(6, 6))
    all_results = {}
    for system_prompt, label in system_prompt_options:
        settings.system_prompt = system_prompt
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
            all_results[system_prompt] = res_list
        except:
            print(f"[WARN] Missing data for system_prompt={system_prompt} for layer={layer}")
    plt.legend()
    plt.xlabel("Multiplier")
    plt.ylabel(f"Probability of answer to A/B question matching {settings.behavior} behavior")
    plt.tight_layout()
    plt.savefig(save_to, format="svg")
    plt.savefig(save_to.replace("svg", "png"), format="png")
    # Save data in all_results used for plotting as .txt
    with open(save_to.replace(".svg", ".txt"), "w") as f:
        for system_prompt, res_list in all_results.items():
            for multiplier, score in res_list:
                f.write(f"{system_prompt}\t{multiplier}\t{score}\n")


def plot_tqa_mmlu_results_for_layer(
    layer: int, multipliers: List[float], settings: SteeringSettings
):
    save_to = os.path.join(
        get_analysis_dir(settings.behavior),
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
    plt.title(f"L{layer}, {settings.behavior}, {settings.type}")
    plt.tight_layout()
    plt.savefig(save_to, format="svg")
    plt.savefig(save_to.replace("svg", "png"), format="png")
    # Save data in res_per_category used for plotting as .txt
    with open(save_to.replace(".svg", ".txt"), "w") as f:
        for category, res_list in res_per_category.items():
            for multiplier, score in res_list:
                f.write(f"{category}\t{multiplier}\t{score}\n")


def plot_open_ended_results(
    layer: int, multipliers: List[float], settings: SteeringSettings
):
    save_to = os.path.join(
        get_analysis_dir(settings.behavior),
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
    plt.ylabel("Average behavioral eval score")
    plt.tight_layout()
    plt.savefig(save_to, format="svg")
    plt.savefig(save_to.replace("svg", "png"), format="png")
    # Save data in res_list used for plotting as .txt
    with open(save_to.replace(".svg", ".txt"), "w") as f:
        for multiplier, score in res_list:
            f.write(f"{multiplier}\t{score}\n")


def plot_ab_data_per_layer(
    layers: List[int], multiplier: float, settings: SteeringSettings
):
    plt.clf()
    plt.figure(figsize=(6, 6))
    save_to = os.path.join(
        get_analysis_dir(settings.behavior),
        f"{settings.make_result_save_suffix()}.svg",
    )
    res = []
    for layer in sorted(layers):
        results = get_data(layer, multiplier, settings)
        avg_key_prob = get_avg_key_prob(results, "answer_matching_behavior")
        res.append(avg_key_prob)
    plt.plot(
        sorted(layers),
        res,
        label=f"Layer {layer}",
        marker="o",
        linestyle="dashed",
        markersize=5,
        linewidth=2.5,
    )
    plt.legend()
    plt.xlabel("Layer")
    plt.ylabel(f"Probability of answer to A/B question matching {settings.behavior} behavior")
    plt.title(f"Multiplier {multiplier}, {settings.behavior}, {settings.type}")
    plt.tight_layout()
    plt.savefig(save_to, format="svg")
    plt.savefig(save_to.replace("svg", "png"), format="png")
    with open(save_to.replace(".svg", ".txt"), "w") as f:
        for layer, score in zip(sorted(layers), res):
            f.write(f"{layer}\t{score}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--multipliers", nargs="+", type=float, required=True)
    parser.add_argument(
        "--behavior",
        type=str,
        required=True,
        choices=ALL_BEHAVIORS,
    )
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=["ab", "open_ended", "truthful_qa", "mmlu"],
    )
    parser.add_argument("--system_prompt", type=str, default=None, choices=["pos", "neg"], required=False)
    parser.add_argument("--override_vector", type=int, default=None)
    parser.add_argument("--override_vector_model", type=str, default=None)
    parser.add_argument("--use_base_model", action="store_true", default=False)
    parser.add_argument("--model_size", type=str, choices=["7b", "13b"], default="7b")
    parser.add_argument("--override_model_weights_path", type=str, default=None)
    
    args = parser.parse_args()

    steering_settings = SteeringSettings()
    steering_settings.type = args.type
    steering_settings.behavior = args.behavior
    steering_settings.system_prompt = args.system_prompt
    steering_settings.override_vector = args.override_vector
    steering_settings.override_vector_model = args.override_vector_model
    steering_settings.use_base_model = args.use_base_model
    steering_settings.model_size = args.model_size
    steering_settings.override_model_weights_path = args.override_model_weights_path

    if steering_settings.type == "ab":
        if len(args.layers) == 1:
            layer = args.layers[0]
            plot_ab_results_for_layer(
                layer, args.multipliers, steering_settings
            )
        else:
            plot_ab_data_per_layer(
                args.layers, args.multipliers[0], steering_settings
            )
    elif steering_settings.type == "open_ended":
        for layer in args.layers:
            plot_open_ended_results(layer, args.multipliers, steering_settings)
    elif steering_settings.type == "truthful_qa" or steering_settings.type == "mmlu":
        for layer in args.layers:
            plot_tqa_mmlu_results_for_layer(layer, args.multipliers, steering_settings)
