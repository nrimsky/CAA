# Sycophancy Activation Steering

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Available commands

```bash
# Format datasets for generating steering vector and testing effect
python make_datasets.py --generate_test_split 0.8 --anthropic_custom_split 0.6 --n_datapoints 1000
# Generate steering vectors and optionally save full activations
python generate_vectors.py --layers 10 11 12 13 14 --save_activations --vector_save_dir vectors --activation_save_dir activations
# Optionally, plot projected activations
python plot_activations.py --activations_pos_file activations/activations_pos_12.pt --activations_neg_file activations/activations_neg_12.pt --fname activations_proj_12.png --title "Activations layer 12"
# Apply steering vectors to model and test effect (pass --out_of_distribution to test on out-of-distribution data)
python prompting_with_steering.py --layers 15 20 25 --multipliers -10 -5 0 5 10 --max_new_tokens 100 --out_of_distribution
```

## Running tests

```bash
pytest
```

## TODO

- [ ] Test all scripts on GPU
- [ ] Add few-shot prompting comparison
- [ ] Adapt for llama-13b and llama-70b