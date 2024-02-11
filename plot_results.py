"""
Plot results from behavioral evaluations under steering.

Example usage:
python plot_results.py --layers $(seq 0 31) --multipliers -1 0 1 --type ab
"""

from tkinter import font
import matplotlib.pyplot as plt
import json
from typing import Dict, Any, List
import os
from collections import defaultdict
import matplotlib.cm as cm
import numpy as np
import argparse
from steering_settings import SteeringSettings
from behaviors import ANALYSIS_PATH, HUMAN_NAMES, get_results_dir, get_analysis_dir, ALL_BEHAVIORS
from utils.helpers import set_plotting_settings

set_plotting_settings()

def get_data(
    layer: int,
    multiplier: int,
    settings: SteeringSettings,
) -> Dict[str, Any]:
    directory = get_results_dir(settings.behavior)
    if settings.type == "open_ended":
        directory = directory.replace("results", os.path.join("results", "open_ended_scores"))
    filenames = settings.filter_result_files_by_suffix(
        directory, layer=layer, multiplier=multiplier
    )
    if len(filenames) > 1:
        print(f"[WARN] >1 filename found for filter {settings}", filenames)
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
            print(f"[WARN] Skipping invalid score: {result}")
    if tot == 0:
        print(f"[WARN] No valid scores found in results")
        return 0.0
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
        ("pos", f"Positive system prompt"),
        ("neg", f"Negative system prompt"),
        (None, f"No system prompt"),
    ]
    settings.system_prompt = None
    save_to = os.path.join(
        get_analysis_dir(settings.behavior),
        f"{settings.make_result_save_suffix(layer=layer)}.png",
    )
    plt.clf()
    plt.figure(figsize=(3.5, 3.5))
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
                linestyle="solid",
                markersize=10,
                linewidth=3,
            )
            all_results[system_prompt] = res_list
        except:
            print(f"[WARN] Missing data for system_prompt={system_prompt} for layer={layer}")
    plt.legend()
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    plt.xlabel("Multiplier")
    plt.ylabel("p(answer matching behavior)")
    plt.xticks(ticks=multipliers, labels=multipliers)
    if (settings.override_vector is None) and (settings.override_vector_model is None) and (settings.override_model_weights_path is None):
        plt.title(f"{HUMAN_NAMES[settings.behavior]} - {settings.get_formatted_model_name()}", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_to, format="png")
    # Save data in all_results used for plotting as .txt
    with open(save_to.replace(".png", ".txt"), "w") as f, open(save_to.replace(".png", ".tex"), "w") as f_tex:
        for system_prompt, res_list in all_results.items():
            for multiplier, score in res_list:
                f.write(f"{system_prompt}\t{multiplier}\t{score}\n")
        if len(all_results) != 3:
            return
        try:
            none_results = dict(all_results[None])[-1], dict(all_results[None])[0], dict(all_results[None])[1]
            pos_results = dict(all_results["pos"])[-1], dict(all_results["pos"])[0], dict(all_results["pos"])[1]
            neg_results = dict(all_results["neg"])[-1], dict(all_results["neg"])[0], dict(all_results["neg"])[1]
            f_tex.write(f"{HUMAN_NAMES[settings.behavior]} & {none_results[0]:.2f} & {none_results[1]:.2f} & {none_results[2]:.2f} & {pos_results[0]:.2f} & {pos_results[1]:.2f} & {pos_results[2]:.2f} & {neg_results[0]:.2f} & {neg_results[1]:.2f} & {neg_results[2]:.2f}")
        except KeyError:
            pass


def plot_tqa_mmlu_results_for_layer(
    layer: int, multipliers: List[float], settings: SteeringSettings
):
    save_to = os.path.join(
        get_analysis_dir(settings.behavior),
        f"{settings.make_result_save_suffix(layer=layer)}.png",
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

    plt.figure(figsize=(10, 5))
    for idx, (category, res_list) in enumerate(sorted(res_per_category.items(), key=lambda x: x[0])):
        x = [idx] * len(res_list)  # Assign a unique x-coordinate for each category
        y = [score for _, score in res_list]  # Extract y-coordinates from the results
        colors = cm.rainbow(np.linspace(0, 1, len(res_list)))  # Assign colors based on the rainbow spectrum
        plt.scatter(x, y, color=colors, s=80)  # Plot the points with the assigned colors

    # Add a legend for the colors with the correct rainbow spectrum cm.rainbow(np.linspace(0, 1, len(multipliers)))
    # Need to ensure dots colored with raindbow rather than default
    for idx, multiplier in enumerate(multipliers):
        plt.scatter([], [], color=cm.rainbow(np.linspace(0, 1, len(multipliers)))[idx], label=f"Multiplier {multiplier}")
    plt.legend(loc="upper left")

    # Final plot adjustments
    plt.xticks(range(len(categories)), categories, rotation=45, ha="right")
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    plt.xlabel("Multiplier")
    plt.ylabel("Probability of correct answer to A/B question")
    if (settings.override_vector is None) and (settings.override_vector_model is None) and (settings.override_model_weights_path is None):
        plt.title(f"Effect of {HUMAN_NAMES[settings.behavior]} CAA on {settings.get_formatted_model_name()} performance")
    plt.tight_layout()
    plt.savefig(save_to, format="png")

    def _format_for_latex_table(x, baseline):
        x_rounded_2dp = round(x, 2)
        baseline_rounded_2dp = round(baseline, 2)
        if x_rounded_2dp < baseline_rounded_2dp:
            return f"\\worse{{{x_rounded_2dp:.2f}}}"
        elif x_rounded_2dp > baseline_rounded_2dp:
            return f"\\better{{{x_rounded_2dp:.2f}}}"
        else:
            return f"\\same{{{x_rounded_2dp:.2f}}}"
        
    def _format_category_name(c):
        # split by _ and capitalize first letter of each word
        return " ".join([word.capitalize() for word in c.split("_")])
    
    # Optionally, save the data used for plotting
    with open(save_to.replace(".png", ".txt"), "w") as f, open(save_to.replace(".png", ".tex"), "w") as f_tex:
        pos_avg = 0
        neg_avg = 0
        no_steering_avg = 0
        for category in sorted(res_per_category.keys()):
            res_list = res_per_category[category]
            res_dict = dict(res_list)
            try:
                no_steering_res = res_dict[0]
                positive_steering_res = res_dict[1]
                negative_steering_res = res_dict[-1]
                pos_avg += positive_steering_res
                neg_avg += negative_steering_res
                no_steering_avg += no_steering_res
                positive_steering_res = _format_for_latex_table(positive_steering_res, no_steering_res)
                negative_steering_res = _format_for_latex_table(negative_steering_res, no_steering_res)
                no_steering_res = f"\\same{{{no_steering_res:.2f}}}"
                f_tex.write(f"{_format_category_name(category)} & {positive_steering_res} & {negative_steering_res} & {no_steering_res} \\\ \n")
            except KeyError:
                pass                
            for multiplier, score in res_list:
                f.write(f"{category}\t{multiplier}\t{score}\n")
        pos_avg /= len(res_per_category)
        neg_avg /= len(res_per_category)
        no_steering_avg /= len(res_per_category)
        pos_avg = _format_for_latex_table(pos_avg, no_steering_avg)
        neg_avg = _format_for_latex_table(neg_avg, no_steering_avg)
        no_steering_avg = f"\\same{{{no_steering_avg:.2f}}}"
        avg_line = f"Average & {pos_avg} & {neg_avg} & {no_steering_avg} \\\ \n"
        f_tex.write(avg_line)


def plot_open_ended_results(
    layer: int, multipliers: List[float], settings: SteeringSettings
):
    save_to = os.path.join(
        get_analysis_dir(settings.behavior),
        f"{settings.make_result_save_suffix(layer=layer)}.png",
    )
    plt.clf()
    plt.figure(figsize=(5, 5))
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
    plt.savefig(save_to, format="png")
    # Save data in res_list used for plotting as .txt
    with open(save_to.replace(".png", ".txt"), "w") as f:
        for multiplier, score in res_list:
            f.write(f"{multiplier}\t{score}\n")


def plot_ab_data_per_layer(
    layers: List[int], multipliers: List[float], settings: SteeringSettings
):
    plt.clf()
    plt.figure(figsize=(10, 4))
    all_results = []
    save_to = os.path.join(
        get_analysis_dir(settings.behavior),
        f"{settings.make_result_save_suffix()}.png",
    )
    for multiplier in multipliers:
        res = []
        for layer in sorted(layers):
            results = get_data(layer, multiplier, settings)
            avg_key_prob = get_avg_key_prob(results, "answer_matching_behavior")
            res.append(avg_key_prob)
        all_results.append(res)
        plt.plot(
            sorted(layers),
            res,
            marker="o",
            linestyle="dashed",
            markersize=10,
            linewidth=4,
            label="Negative steering" if multiplier < 0 else "Positive steering",
        )
    # use % formatting for y axis
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    if (settings.override_vector is None) and (settings.override_vector_model is None) and (settings.override_model_weights_path is None):
        plt.title(f"{HUMAN_NAMES[settings.behavior]} CAA, {settings.get_formatted_model_name()}")
    plt.xlabel("Layer")
    plt.ylabel("Probability of answer matching behavior")
    plt.xticks(ticks=sorted(layers), labels=sorted(layers))
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_to, format="png")
    with open(save_to.replace(".png", ".txt"), "w") as f:
        for layer in sorted(layers):
            f.write(f"{layer}\t")
            for idx, multiplier in enumerate(multipliers):
                f.write(f"{all_results[idx]}\t")
            f.write("\n")

def plot_effect_on_behaviors(
    layer: int, multipliers: List[int], behaviors: List[str], settings: SteeringSettings, title: str = None   
):
    plt.clf()
    plt.figure(figsize=(3, 3))
    save_to = os.path.join(
        ANALYSIS_PATH,
        f"{settings.make_result_save_suffix(layer=layer)}.png",
    )
    all_results = []
    for behavior in behaviors:
        results = []
        for mult in multipliers:
            settings.behavior = behavior
            data = get_data(layer, mult, settings)
            if settings.type == "open_ended":
                avg_score = get_avg_score(data)
                results.append(avg_score)
            elif settings.type == "ab":
                avg_key_prob = get_avg_key_prob(data, "answer_matching_behavior")
                results.append(avg_key_prob * 100)
            else:
                avg_key_prob = get_avg_key_prob(data, "correct")
                results.append(avg_key_prob * 100)
        all_results.append(results)

    for idx, behavior in enumerate(behaviors):
        plt.plot(
            multipliers,
            all_results[idx],
            marker="o",
            linestyle="solid",
            markersize=10,
            linewidth=3,
            label=HUMAN_NAMES[behavior],
        )
    plt.xticks(ticks=multipliers, labels=multipliers)
    if title is not None:
        plt.title(title)
    plt.xlabel("Steering vector multiplier")
    ylabel = "p(answer matching behavior)"
    if settings.type == "open_ended":
        ylabel = "Mean behavioral score (/10)"
    elif settings.type == "mmlu":
        ylabel = "p(correct answer to A/B question)"
    elif settings.type == "truthful_qa":
        ylabel = "p(correct answer to A/B question)"
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_to, format="png")
    plt.savefig(save_to.replace("png", "svg"), format="svg")
    with open(save_to.replace(".png", ".txt"), "w") as f:
        for mult in multipliers:
            f.write(f"{mult}\t")
            for idx, behavior in enumerate(behaviors):
                f.write(f"{all_results[idx]}\t")
            f.write("\n")

def plot_layer_sweeps(
    layers: List[int], behaviors: List[str], settings: SteeringSettings, title: str = None
):
    plt.clf()
    plt.figure(figsize=(5, 3))
    all_results = []
    save_to = os.path.join(
        ANALYSIS_PATH,
        f"LAYER_SWEEPS_{settings.make_result_save_suffix()}.png",
    )
    for behavior in behaviors:
        if "coordinate" in behavior:
            continue
        settings.behavior = behavior
        pos_per_layer = []
        neg_per_layer = []
        for layer in sorted(layers):
            base_res = get_avg_key_prob(get_data(layer, 0, settings), "answer_matching_behavior")
            pos_res = get_avg_key_prob(get_data(layer, 1, settings), "answer_matching_behavior") - base_res
            neg_res = get_avg_key_prob(get_data(layer, -1, settings), "answer_matching_behavior") - base_res
            pos_per_layer.append(pos_res)
            neg_per_layer.append(neg_res)
        all_results.append((pos_per_layer, neg_per_layer))
        plt.plot(
            sorted(layers),
            pos_per_layer,
            linestyle="solid",
            linewidth=2,
            color="#377eb8",
        )
        plt.plot(
            sorted(layers),
            neg_per_layer,
            linestyle="solid",
            linewidth=2,
            color="#ff7f00",
        )

    plt.plot(
        [],
        [],
        linestyle="solid",
        linewidth=2,
        color="#377eb8",
        label="Positive steering",
    )
    plt.plot(
        [],
        [],
        linestyle="solid",
        linewidth=2,
        color="#ff7f00",
        label="Negative steering",
    )

    # use % formatting for y axis
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    plt.xlabel("Layer")
    plt.ylabel("$\Delta$ p(answer matching behavior)")
    if not title:
        plt.title(f"Per-layer CAA effect: {settings.get_formatted_model_name()}")
    else:
        plt.title(title)
    plt.xticks(ticks=sorted(layers)[::5], labels=sorted(layers)[::5])
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_to, format="png")

def steering_settings_from_args(args, behavior: str) -> SteeringSettings:
    steering_settings = SteeringSettings()
    steering_settings.type = args.type
    steering_settings.behavior = behavior
    steering_settings.override_vector = args.override_vector
    steering_settings.override_vector_model = args.override_vector_model
    steering_settings.use_base_model = args.use_base_model
    steering_settings.model_size = args.model_size
    steering_settings.override_model_weights_path = args.override_model_weights_path
    return steering_settings

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--multipliers", nargs="+", type=float, required=True)
    parser.add_argument("--title", type=str, required=False, default=None)
    parser.add_argument(
        "--behaviors",
        type=str,
        nargs="+",
        default=ALL_BEHAVIORS,
    )
    parser.add_argument(
        "--type",
        type=str,
        default="ab",
        choices=["ab", "open_ended", "truthful_qa", "mmlu"],
    )
    parser.add_argument("--override_vector", type=int, default=None)
    parser.add_argument("--override_vector_model", type=str, default=None)
    parser.add_argument("--use_base_model", action="store_true", default=False)
    parser.add_argument("--model_size", type=str, choices=["7b", "13b"], default="7b")
    parser.add_argument("--override_model_weights_path", type=str, default=None)
    
    args = parser.parse_args()

    steering_settings = steering_settings_from_args(args, args.behaviors[0])

    if steering_settings.type == "ab":
        plot_layer_sweeps(args.layers, args.behaviors, steering_settings, args.title)

    if len(args.layers) == 1 and steering_settings.type != "truthful_qa":
        plot_effect_on_behaviors(args.layers[0], args.multipliers, args.behaviors, steering_settings, args.title)

    for behavior in args.behaviors:
        steering_settings = steering_settings_from_args(args, behavior)
        if steering_settings.type == "ab":
            if len(args.layers) > 1 and 1 in args.multipliers and -1 in args.multipliers:
                plot_ab_data_per_layer(
                    args.layers, [1, -1], steering_settings
                )
            if len(args.layers) == 1:
                plot_ab_results_for_layer(args.layers[0], args.multipliers, steering_settings)
        elif steering_settings.type == "open_ended":
            for layer in args.layers:
                plot_open_ended_results(layer, args.multipliers, steering_settings)
        elif steering_settings.type == "truthful_qa" or steering_settings.type == "mmlu":
            for layer in args.layers:
                plot_tqa_mmlu_results_for_layer(layer, args.multipliers, steering_settings)
