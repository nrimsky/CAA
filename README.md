# Sycophancy Activation Steering

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then create a `.env` file with the following variables (see `.env.example`):

```bash
HF_TOKEN=huggingface_token_with_access_to_llama2
```

## Available commands

```bash
# Format datasets for generating steering vector and testing effect
python make_datasets.py --generate_test_split 0.8 --anthropic_custom_split 0.6 --n_datapoints 1000 --n_tqa_datapoints 200
# Generate steering vectors and optionally save full activations
python generate_vectors.py --layers 15 20 25 --save_activations --vector_save_dir vectors --activation_save_dir activations
# Optionally, plot projected activations
python plot_activations.py --activations_pos_file activations/activations_pos_15.pt --activations_neg_file activations/activations_neg_15.pt --fname activations_proj_15.png --title "Activations layer 15"
# Apply steering vectors to model and test effect (pass --out_of_distribution to test on out-of-distribution data)
python prompting_with_steering.py --layers 15 20 25 --multipliers -5 -2 -1 0 1 2 5 --max_new_tokens 80 --out_of_distribution
```

## Running tests

I have added a few unit tests for some of the utility functions. To run them, simply run:

```bash
pytest
```

## TODO

- [ ] Add TruthfulQA eval to `prompting_with_steering.py`
- [ ] Add few-shot prompting comparison
- [ ] Adapt for llama-13b and llama-70b
- [ ] Add more unit tests
- [ ] Add Jupyter notebook for examining and visualizing vector similarity between layers and tokens