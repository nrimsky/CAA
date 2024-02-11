"""
Usage: python analyze_vectors.py
"""

import os
from matplotlib.pylab import f
import torch as t
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from behaviors import ALL_BEHAVIORS, get_analysis_dir, HUMAN_NAMES, get_steering_vector, ANALYSIS_PATH
from utils.helpers import get_model_path, model_name_format, set_plotting_settings
from tqdm import tqdm

set_plotting_settings()

def get_caa_info(behavior: str, model_size: str, is_base: bool):
    all_vectors = []
    n_layers = 36 if "13" in model_size else 32
    model_path = get_model_path(model_size, is_base)
    for layer in range(n_layers):
        all_vectors.append(get_steering_vector(behavior, layer, model_path))
    return {
        "vectors": all_vectors,
        "n_layers": n_layers,
        "model_name": model_name_format(model_path),
    }

def plot_per_layer_similarities(model_size: str, is_base: bool, behavior: str):
    analysis_dir = get_analysis_dir(behavior)
    caa_info = get_caa_info(behavior, model_size, is_base)
    all_vectors = caa_info["vectors"]
    n_layers = caa_info["n_layers"]
    model_name = caa_info["model_name"]
    matrix = np.zeros((n_layers, n_layers))
    for layer1 in range(n_layers):
        for layer2 in range(n_layers):
            cosine_sim = t.nn.functional.cosine_similarity(all_vectors[layer1], all_vectors[layer2], dim=0).item()
            matrix[layer1, layer2] = cosine_sim
    plt.figure(figsize=(3, 3))
    sns.heatmap(matrix, annot=False, cmap='coolwarm')
    # Set ticks for every 5th layer
    plt.xticks(list(range(n_layers))[::5], list(range(n_layers))[::5])
    plt.yticks(list(range(n_layers))[::5], list(range(n_layers))[::5])
    plt.title(f"Layer similarity, {model_name}", fontsize=11)
    plt.savefig(os.path.join(analysis_dir, f"cosine_similarities_{model_name.replace(' ', '_')}_{behavior}.svg"), format='svg')
    plt.close()

def plot_base_chat_similarities():
    plt.figure(figsize=(5, 3))
    for behavior in ALL_BEHAVIORS:
        base_caa_info = get_caa_info(behavior, "7b", True)
        chat_caa_info = get_caa_info(behavior, "7b", False)
        vectors_base = base_caa_info["vectors"]
        vectors_chat = chat_caa_info["vectors"]
        cos_sims = []
        for layer in range(base_caa_info["n_layers"]):
            cos_sim = t.nn.functional.cosine_similarity(vectors_base[layer], vectors_chat[layer], dim=0).item()
            cos_sims.append(cos_sim)
        plt.plot(list(range(base_caa_info["n_layers"])), cos_sims, label=HUMAN_NAMES[behavior], linestyle="solid", linewidth=2)
    plt.xlabel("Layer")
    plt.ylabel("Cosine Similarity")
    plt.title("Base vs. Chat model vector similarity", fontsize=12)
    # legend in bottom right
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_PATH, "base_chat_similarities.png"), format='png')
    plt.close()

if __name__ == "__main__":
    for behavior in tqdm(ALL_BEHAVIORS):
        plot_per_layer_similarities("7b", True, behavior)
        plot_per_layer_similarities("7b", False, behavior)
        plot_per_layer_similarities("13b", False, behavior)
    plot_base_chat_similarities()
