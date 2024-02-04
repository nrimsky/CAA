import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from behaviors import get_results_dir, ALL_BEHAVIORS
import glob
import json


def prepare_for_scoring():
    current_path = os.path.dirname(os.path.realpath(__file__))
    open_ended_scores_dir = os.path.join(current_path, "open_ended_scores")
    if not os.path.exists(open_ended_scores_dir):
        os.makedirs(open_ended_scores_dir)
    for behavior in ALL_BEHAVIORS:
        results_dir = get_results_dir(behavior)
        open_ended_results = glob.glob(f"{results_dir}/*open_ended*")
        copy_dir = os.path.join(open_ended_scores_dir, behavior)
        if not os.path.exists(copy_dir):
            os.makedirs(copy_dir)
        for file in open_ended_results:
            new_save = os.path.join(copy_dir, os.path.basename(file))
            if os.path.exists(new_save):
                print(f"Skipping {file} because it already exists")
                continue
            with open(file, "r") as f:
                data = json.load(f)
            with open(os.path.join(copy_dir, os.path.basename(file)), "w") as f:
                print(f"Writing {file} to {new_save} for scoring")
                data = [{**d, "score": 0} for d in data]
                json.dump(data, f, indent=4)

if __name__ == "__main__":
    prepare_for_scoring()