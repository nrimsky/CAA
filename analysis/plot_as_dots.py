import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob


def plot_as_dots(filename):
    # Load and parse the TSV data
    data = pd.read_csv(filename, sep="\t", header=None)

    num_columns = len(data.columns)

    if num_columns == 3:
        data.columns = ["Category", "Multiplier", "Probability"]
    else:
        print("Not 3 columns - skipping")
        return

    # Create the updated plot with the specified changes
    plt.figure(figsize=(12, 8))

    # Create a color map for the unique multipliers
    unique_multipliers = sorted(data["Multiplier"].unique())
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_multipliers)))
    color_map = {multiplier: color for multiplier, color in zip(unique_multipliers, colors)}

    # Plot points and lines for each category
    for category in data["Category"].unique():
        category_data = data[data["Category"] == category]
        for idx, row in category_data.iterrows():
            plt.scatter(category, row["Probability"], color=color_map[row["Multiplier"]], s=100, marker='o')
        plt.plot([category]*len(category_data), category_data["Probability"], linestyle='-', color='grey', alpha=0.5)

    # Add the discrete legend
    for multiplier, color in color_map.items():
        plt.plot([], [], 'o', color=color, label=f"{multiplier}", markersize=8)

    plt.legend(loc="upper left", title="Multipliers")

    # Set plot details
    # plt.title("Probability of Correct Answer on TruthfulQA Category by Multiplier")
    # plt.xlabel("Category")
    # plt.ylabel("Probability of Correct Answer")
    plt.xticks(rotation=45)
    # Make sure every x tick appears
    # if x axis is categorical
    if data["Category"].dtype == object:
        plt.xticks(np.arange(len(data["Category"].unique())), data["Category"].unique())
    # if x axis is numerical, plot corresponding to correct numbers
    else:
        start = data["Category"].min()
        end = data["Category"].max()
        plt.xticks(np.arange(start, end+1, 1))

    # Display the plot
    plt.tight_layout()
    plt.savefig(f"{filename.split('.')[-2] + '_DOTS'}.png", format="png")
    plt.savefig(f"{filename.split('.')[-2] + '_DOTS'}.svg", format="svg")

if __name__ == "__main__":
    txt_files = glob("*.txt")
    for filename in txt_files:
        plot_as_dots(filename)