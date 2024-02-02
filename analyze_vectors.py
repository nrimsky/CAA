"""
Usage: python analyze_vectors.py
"""

import os
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
from behaviors import ALL_BEHAVIORS, get_vector_dir, get_analysis_dir

plt.style.use('seaborn')
params = {
    "ytick.color" : "black",
    "xtick.color" : "black",
    "axes.labelcolor" : "black",
    "axes.edgecolor" : "black",
    "font.family" : "serif",
    "font.size": 11,
    "figure.autolayout": True,
}
plt.rcParams.update(params)

"""
Utils
"""

def model_label(model_name):
    if "chat" in model_name:
        return "Chat"
    return "Base"

# Load vectors
def load_vectors(behavior):
    vectors = {}
    directory = get_vector_dir(behavior)
    for filename in os.listdir(directory):
        if filename.endswith('.pt'):
            path = os.path.join(directory, filename)
            parts = filename[:-3].split('_')
            layer = int(parts[-2])
            model_name = parts[-1]
            if model_name not in vectors:
                vectors[model_name] = {}
            v = torch.load(path)
            vectors[model_name][layer] = (v - v.mean()) / v.std()
    return vectors

"""
Script
"""

behavior = input(f"Enter behavior ({', '.join(ALL_BEHAVIORS)}): ")
assert behavior in ALL_BEHAVIORS, f"Behavior must be one of {ALL_BEHAVIORS}"

vectors = load_vectors(behavior)
analysis_dir = get_analysis_dir(behavior)
if not os.path.exists(analysis_dir):
    os.makedirs(analysis_dir)

# Initialize PCA
pca = PCA(n_components=2)

# Comparing vectors from different layers of the same model
for model_name, layers in vectors.items():
    keys = list(layers.keys())
    keys.sort()
    
    # Cosine Similarity
    matrix = np.zeros((len(keys), len(keys)))
    for i, layer1 in enumerate(keys):
        for j, layer2 in enumerate(keys):
            cosine_sim = cosine_similarity(layers[layer1].reshape(1, -1), layers[layer2].reshape(1, -1))
            matrix[i, j] = cosine_sim
    plt.figure(figsize=(5, 5))
    sns.heatmap(matrix, annot=False, xticklabels=keys, yticklabels=keys, cmap='coolwarm')
    plt.title(f"Layer vector similarity: {model_name}")
    plt.savefig(os.path.join(analysis_dir, f"cosine_similarities_{model_name}.png"), format='png')
    plt.savefig(os.path.join(analysis_dir, f"cosine_similarities_{model_name}.svg"), format='svg')

    # PCA Projections
    data = [layers[layer].numpy() for layer in keys]
    projections = pca.fit_transform(data)
    plt.figure(figsize=(5, 5))
    plt.scatter(projections[:, 0], projections[:, 1], c=keys, cmap='viridis')
    plt.colorbar().set_label('Layer Number')
    for i, layer in enumerate(keys):
        plt.annotate(layer, (projections[i, 0], projections[i, 1]))
    plt.title(f"Layer vector PCA: {model_name}")
    plt.savefig(os.path.join(analysis_dir, f"pca_layers_{model_name}.png"), format='png')
    plt.savefig(os.path.join(analysis_dir, f"pca_layers_{model_name}.svg"), format='svg')


# Remove 'Llama-2-13b-chat-hf' from vectors
del vectors['Llama-2-13b-chat-hf']

# Comparing vectors from the same layer but different models
common_layers = sorted(list(set(next(iter(vectors.values())).keys())))  # Sorted common layers
model_names = list(vectors.keys())

# Determine grid size for subplots
num_layers = len(common_layers)
cols = 3  # number of columns
rows = num_layers // cols
rows += num_layers % cols
position = range(1, num_layers + 1)

fig_cosine, axarr_cosine = plt.subplots(rows, cols, figsize=(8, 4 * rows))

# Max and min values for consistent color scale across subplots
vmin = float('inf')
vmax = float('-inf')

# Calculate max and min first for consistent color scaling
for layer in common_layers:
    matrix = np.zeros((len(model_names), len(model_names)))
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            try:
                cosine_sim = cosine_similarity(vectors[model1][layer].reshape(1, -1), vectors[model2][layer].reshape(1, -1))
                matrix[i, j] = cosine_sim
                vmin = min(vmin, cosine_sim)
                vmax = max(vmax, cosine_sim)
            except KeyError:
                pass

# Cosine Similarity
for layer, pos in zip(common_layers, position):
    row = (pos - 1) // cols
    col = (pos - 1) % cols
    ax = axarr_cosine[row, col]
    matrix = np.zeros((len(model_names), len(model_names)))
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            try:
                cosine_sim = cosine_similarity(vectors[model1][layer].reshape(1, -1), vectors[model2][layer].reshape(1, -1))
                matrix[i, j] = cosine_sim
            except KeyError:
                pass
    sns.heatmap(matrix, ax=ax, annot=False, xticklabels=model_names, yticklabels=model_names, cmap='coolwarm', vmin=vmin, vmax=vmax)
    ax.set_title(f"Layer: {layer}")

# Remove unused subplots
for pos in range(num_layers+1, (rows*cols)+1):
    row = (pos - 1) // cols
    col = (pos - 1) % cols
    fig_cosine.delaxes(axarr_cosine[row, col])

# Add a single colorbar
cbar_ax = fig_cosine.add_axes([0.92, 0.12, 0.02, 0.76])  # [left, bottom, width, height]
norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
cb1 = matplotlib.colorbar.ColorbarBase(cbar_ax, cmap='coolwarm', norm=norm, orientation='vertical')

# Save plots
fig_cosine.suptitle("Model vector similarity")
fig_cosine.savefig(os.path.join(analysis_dir, f"cosine_similarities_all_models.png"), format='png', bbox_inches='tight')
fig_cosine.savefig(os.path.join(analysis_dir, f"cosine_similarities_all_models.svg"), format='svg', bbox_inches='tight')

# Plot PCA of all layers over all models on a single plot with layer, model labels
plt.clf()
data = []
labels = []
for layer in common_layers:
    for model_name in model_names:
        data.append(vectors[model_name][layer].numpy())
        labels.append(model_label(model_name))
data = np.array(data)
projections = pca.fit_transform(data)

plt.figure(figsize=(5, 5))

# Separate chat and base data
chat_data = np.array([projections[i] for i, label in enumerate(labels) if label == "Chat"])
base_data = np.array([projections[i] for i, label in enumerate(labels) if label == "Base"])

# Plot chat models with crosses
plt.scatter(chat_data[:, 0], chat_data[:, 1], c='blue', marker='x', label='Chat')

# Plot base models with circles
plt.scatter(base_data[:, 0], base_data[:, 1], c='red', marker='o', label='Base')

# Colorbar and legend
plt.legend()

# Annotate each point
for i, layer in enumerate(common_layers):
    for j, model_name in enumerate(model_names):
        plt.annotate(str(layer), (projections[i*len(model_names)+j, 0], projections[i*len(model_names)+j, 1]))

plt.title(f"Layer vector PCA: {', '.join(model_names)}")
plt.savefig(os.path.join(analysis_dir, f"pca_layers_all_models.png"), format='png')
plt.savefig(os.path.join(analysis_dir, f"pca_layers_all_models.svg"), format='svg')

# Plotting cosine similarity between the first 2 models per layer number
model1_name, model2_name = model_names[0], model_names[1]
cosine_sims_per_layer = []

for layer in common_layers:
    cosine_sim = cosine_similarity(vectors[model1_name][layer].reshape(1, -1), vectors[model2_name][layer].reshape(1, -1))[0][0]
    cosine_sims_per_layer.append(cosine_sim)

plt.figure(figsize=(5, 5))
plt.plot(common_layers, cosine_sims_per_layer, marker='o', linestyle='-')
plt.xlabel("Layer Number")
plt.ylabel("Cosine Similarity")
plt.title(f"Similarity between {model1_name} and {model2_name}")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(analysis_dir, f"cosine_similarity_{model1_name}_{model2_name}.png"), format='png')
plt.savefig(os.path.join(analysis_dir, f"cosine_similarity_{model1_name}_{model2_name}.svg"), format='svg')