"""
Script to plot PCA of constrastive activations

Usage:
python plot_activations.py --behaviors sycophancy --layers 9 10 11 --use_base_model --model_size 7b
"""

import json
import torch as t
import os
from matplotlib import pyplot as plt
import argparse
from sklearn.decomposition import PCA
from behaviors import HUMAN_NAMES, get_activations_path, get_ab_data_path, get_analysis_dir, ALL_BEHAVIORS
from utils.helpers import get_model_path, set_plotting_settings
from tqdm import tqdm

DATASET_FILE = os.path.join("preprocessed_data", "generate_dataset.json")

set_plotting_settings()

def save_activation_projection_pca(behavior: str, layer: int, model_name_path: str):
    title = f"{HUMAN_NAMES[behavior]}, layer {layer}"
    fname = f"pca_{behavior}_layer_{layer}.png"
    save_dir = os.path.join(get_analysis_dir(behavior), "pca")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Loading activations
    activations_pos = t.load(
        get_activations_path(behavior, layer, model_name_path, "pos")
    )
    activations_neg = t.load(
        get_activations_path(behavior, layer, model_name_path, "neg")
    )

    # Getting letters
    with open(get_ab_data_path(behavior), "r") as f:
        data = json.load(f)

    letters_pos = [item["answer_matching_behavior"][1] for item in data]
    letters_neg = [item["answer_not_matching_behavior"][1] for item in data]

    plt.clf()
    plt.figure(figsize=(4, 4))
    activations = t.cat([activations_pos, activations_neg], dim=0)
    activations_np = activations.cpu().numpy()

    # PCA projection
    pca = PCA(n_components=2)
    projected_activations = pca.fit_transform(activations_np)

    # Splitting back into activations1 and activations2
    activations_pos_projected = projected_activations[: activations_pos.shape[0]]
    activations_neg_projected = projected_activations[activations_pos.shape[0] :]

    # Visualization
    for i, (x, y) in enumerate(activations_pos_projected):
        if letters_pos[i] == "A":
            plt.scatter(x, y, color="blue", marker="o", alpha=0.4)
        elif letters_pos[i] == "B":
            plt.scatter(x, y, color="blue", marker="x", alpha=0.4)

    for i, (x, y) in enumerate(activations_neg_projected):
        if letters_neg[i] == "A":
            plt.scatter(x, y, color="red", marker="o", alpha=0.4)
        elif letters_neg[i] == "B":
            plt.scatter(x, y, color="red", marker="x", alpha=0.4)

    # Adding the legend
    scatter1 = plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="blue",
        markersize=10,
        label=f"pos {HUMAN_NAMES[behavior]} - A",
    )
    scatter2 = plt.Line2D(
        [0],
        [0],
        marker="x",
        color="blue",
        markerfacecolor="blue",
        markersize=10,
        label=f"pos {HUMAN_NAMES[behavior]} - B",
    )
    scatter3 = plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="red",
        markersize=10,
        label=f"neg {HUMAN_NAMES[behavior]} - A",
    )
    scatter4 = plt.Line2D(
        [0],
        [0],
        marker="x",
        color="red",
        markerfacecolor="red",
        markersize=10,
        label=f"neg {HUMAN_NAMES[behavior]} - B",
    )

    plt.legend(handles=[scatter1, scatter2, scatter3, scatter4])
    plt.title(title)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.savefig(os.path.join(save_dir, fname), format="png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--behaviors",
        nargs="+",
        type=str,
        default=ALL_BEHAVIORS,
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        required=True,
    )
    parser.add_argument("--use_base_model", action="store_true", default=False)
    parser.add_argument("--model_size", type=str, choices=["7b", "13b"], default="7b")
    args = parser.parse_args()
    model_name_path = get_model_path(args.model_size, args.use_base_model)
    args = parser.parse_args()
    for behavior in args.behaviors:
        print(f"plotting {behavior} activations PCA")
        for layer in tqdm(args.layers):
            save_activation_projection_pca(
                behavior,
                layer,
                model_name_path,
            )
