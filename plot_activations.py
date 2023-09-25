"""
Script to plot activations of sycophantic and non-sycophantic responses

Usage:
python plot_activations.py --activations_pos_file activations/activations_pos_12.pt --activations_neg_file activations/activations_neg_12.pt --fname activations_proj_12.png --title "Activations layer 12"
"""

import json
import torch as t
import os
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import argparse

DATASET_FILE = os.path.join("preprocessed_data", "generate_dataset.json")

def save_activation_projection_tsne(
    activations_pos_file, activations_neg_file, fname, title, dataset_file
):
    """
    activations_pos: path to saved n_samples x residual_dim tensor
    activations_neg: path to saved n_samples x residual_dim tensor
    fname: filename to save visualization to
    title: title of plot
    dataset_file: path to dataset file used to generate activations_pos and activations_neg (for getting letters)

    Projects to n_samples x 2 dim tensor using t-SNE (over the full dataset of both activations 1 and 2) and saves visualization
    Colors projected activations_pos as blue and projected activations_neg as red
    Marks A activations as circles and B activations as crosses
    """
    # Loading activations
    activations_pos = t.load(activations_pos_file)
    activations_neg = t.load(activations_neg_file)

    # Getting letters
    with open(dataset_file, "r") as f:
        data = json.load(f)

    letters_pos = [item["answer_matching_behavior"][1] for item in data]
    letters_neg = [item["answer_not_matching_behavior"][1] for item in data]

    plt.clf()
    activations = t.cat([activations_pos, activations_neg], dim=0)
    activations_np = activations.cpu().numpy()

    # t-SNE transformation
    tsne = TSNE(n_components=2)
    projected_activations = tsne.fit_transform(activations_np)

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
        label="Sycophantic activations - A",
    )
    scatter2 = plt.Line2D(
        [0],
        [0],
        marker="x",
        color="blue",
        markerfacecolor="blue",
        markersize=10,
        label="Sycophantic activations - B",
    )
    scatter3 = plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="red",
        markersize=10,
        label="Non sycophantic activations - A",
    )
    scatter4 = plt.Line2D(
        [0],
        [0],
        marker="x",
        color="red",
        markerfacecolor="red",
        markersize=10,
        label="Non sycophantic activations - B",
    )

    plt.legend(handles=[scatter1, scatter2, scatter3, scatter4])
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.savefig(fname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations_pos_file", type=str, default="activations_pos.pt")
    parser.add_argument("--activations_neg_file", type=str, default="activations_neg.pt")
    parser.add_argument("--fname", type=str, default="activations_pos_neg.png")
    parser.add_argument("--title", type=str, default="Activations of sycophantic and non-sycophantic responses")
    parser.add_argument("--dataset_file", type=str, default=DATASET_FILE)
    args = parser.parse_args()
    save_activation_projection_tsne(args.activations_pos_file, args.activations_neg_file, args.fname, args.title, args.dataset_file)