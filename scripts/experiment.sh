# Execute full experiment - generates results for all datasets suitable for further analysis

# Usage (from repo root): 
# chmod +x ./scripts/experiment.sh
# ./scripts/experiment.sh

# Create venv and install requirements if not already done
if [ ! -d "venv" ]; then
    echo "Creating venv"
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
fi

# Generate datasets
python make_datasets.py --generate_test_split 0.8 --anthropic_custom_split 0.6 --n_datapoints 1200 --n_tqa_datapoints 200

# Generate steering vector for layer 15
python generate_vectors.py --layers 15 --save_activations

# Plot projected activations
python plot_activations.py --activations_pos_file activations/activations_pos_15.pt --activations_neg_file activations/activations_neg_15.pt --fname activations_proj_15.png --title "Activations layer 15"

# In-distribution tests
python prompting_with_steering.py --type in_distribution --layers 15 --multipliers -2 -1 -0.5 0 0.5 1 2 --few_shot unbiased
python prompting_with_steering.py --type in_distribution --layers 15 --multipliers -2 -1 -0.5 0 0.5 1 2 --few_shot positive
python prompting_with_steering.py --type in_distribution --layers 15 --multipliers -2 -1 -0.5 0 0.5 1 2 --few_shot negative

# Out-of-distribution tests
python prompting_with_steering.py --type out_of_distribution --layers 15 --multipliers -2 -1 -0.5 0 0.5 1 2 --max_new_tokens 100

# TruthfulQA tests
python prompting_with_steering.py --type truthful_qa --layers 15 --multipliers -2 -1 -0.5 0 0.5 1 2 --few_shot unbiased


