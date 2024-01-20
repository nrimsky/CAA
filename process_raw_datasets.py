import os
import json
import random
import behaviors

N_GENERATE = 400
N_TEST = 50


def make_generate_test_datasets(behavior):
    read_path = behaviors.get_raw_data_path(behavior)
    write_generate_path = behaviors.get_ab_data_path(behavior)
    write_test_path = behaviors.get_ab_data_path(behavior, test=True)
    write_open_ended_test_path = behaviors.get_open_ended_data_path(behavior)
    # make subdirectories if they don't exist
    os.makedirs(os.path.dirname(write_generate_path), exist_ok=True)
    os.makedirs(os.path.dirname(write_test_path), exist_ok=True)
    with open(read_path, "r") as f:
        data = json.load(f)
    cleaned_data = []
    for item in data:
        cleaned_data.append(
            {
                "question": item["question"].replace("Question:", "").replace("Answer:", "").strip(),
                "answer_matching_behavior": item["answer_matching_behavior"].strip(),
                "answer_not_matching_behavior": item["answer_not_matching_behavior"].strip(),
            }
        )
    random.shuffle(cleaned_data)
    print(len(cleaned_data))
    generate_data = cleaned_data[:N_GENERATE]
    test_data = cleaned_data[N_GENERATE : N_GENERATE + N_TEST]
    with open(write_generate_path, "w") as f:
        json.dump(generate_data, f, indent=4)
    with open(write_test_path, "w") as f:
        json.dump(test_data, f, indent=4)
    open_ended_test = []
    for item in test_data:
        open_ended_test.append(
            {
                "question": item["question"].split("\n")[0].strip(),
            }
        )
    if behavior != "sycophancy":
        # Use custom open-ended test for sycophancy
        with open(write_open_ended_test_path, "w") as f:
            json.dump(open_ended_test, f, indent=4)

def main():
    for behavior in behaviors.ALL_BEHAVIORS:
        make_generate_test_datasets(behavior)

if __name__ == "__main__":
    main()
