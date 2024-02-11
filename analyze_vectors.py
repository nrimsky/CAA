"""
Usage: python analyze_vectors.py
"""

import os
import torch as t
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
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
    plt.figure(figsize=(5, 5))
    sns.heatmap(matrix, annot=False, cmap='coolwarm')
    # Set ticks for every 5th layer
    plt.xticks(list(range(n_layers))[::5], list(range(n_layers))[::5])
    plt.yticks(list(range(n_layers))[::5], list(range(n_layers))[::5])
    plt.title(f"Inter-layer similarity, {model_name}")
    plt.savefig(os.path.join(analysis_dir, f"cosine_similarities_{model_name.replace(' ', '_')}_{behavior}.png"), format='png')
    plt.close()

def plot_base_chat_similarities():
    plt.figure(figsize=(8, 4))
    for behavior in ALL_BEHAVIORS:
        base_caa_info = get_caa_info(behavior, "7b", True)
        chat_caa_info = get_caa_info(behavior, "7b", False)
        vectors_base = base_caa_info["vectors"]
        vectors_chat = chat_caa_info["vectors"]
        cos_sims = []
        for layer in range(base_caa_info["n_layers"]):
            cos_sim = t.nn.functional.cosine_similarity(vectors_base[layer], vectors_chat[layer], dim=0).item()
            cos_sims.append(cos_sim)
        plt.plot(list(range(base_caa_info["n_layers"])), cos_sims, label=HUMAN_NAMES[behavior])
    plt.xlabel("Layer")
    plt.ylabel("Cosine Similarity")
    plt.title("Steering vector similarity between Llama 2 base and chat")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_PATH, "base_chat_similarities.png"), format='png')
    plt.close()
        
def plot_pca_of_all_vectors():
    """
    plot pca of all vectors in llama 2 7b chat
    normalize vectors before pca
    """
    all_vectors = []
    n_layers = 32
    for behavior in ALL_BEHAVIORS:
        caa_info = get_caa_info(behavior, "7b", False)
        all_vectors.extend(caa_info["vectors"])
    all_vectors = t.stack(all_vectors)
    # normalize vectors for pca (mean 0, std 1)
    all_vectors = (all_vectors - all_vectors.mean(dim=0)) / all_vectors.std(dim=0)
    pca = PCA(n_components=2)
    pca.fit(all_vectors)
    pca_vectors = pca.transform(all_vectors)
    plt.figure(figsize=(5, 5))
    for i, behavior in enumerate(ALL_BEHAVIORS):
        start = i * n_layers
        end = start + n_layers
        plt.scatter(pca_vectors[start:end, 0], pca_vectors[start:end, 1], label=HUMAN_NAMES[behavior])
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of all steering vectors")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_PATH, "pca_all_vectors.png"), format='png')
    plt.close()

if __name__ == "__main__":
    for behavior in ALL_BEHAVIORS:
        plot_per_layer_similarities("7b", True, behavior)
        plot_per_layer_similarities("7b", False, behavior)
        plot_per_layer_similarities("13b", False, behavior)
    plot_base_chat_similarities()
    plot_pca_of_all_vectors()
