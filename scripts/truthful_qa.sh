# Assess the effect of activation steering on TruthfulQA

# Create venv and install requirements if not already done
if [ ! -d "venv" ]; then
    echo "Creating venv"
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Generate datasets
echo "Generating datasets"
python make_datasets.py --generate_test_split 0.8 --anthropic_custom_split 0.6 --n_datapoints 1200

# Generate steering vector for layer 16
echo "Generating steering vectors"
python generate_vectors.py --layers 16

# TruthfulQA tests
echo "TruthfulQA A/B question tests"
python prompting_with_steering.py --type truthful_qa --layers 16 --multipliers -1.5 -1 -0.5 0 0.5 1 1.5 --few_shot none

# Plot results
echo "Plotting results"
python analysis/plot_results.py --multipliers -1.5 -1 -0.5 0 0.5 1 1.5 --type truthful_qa --layers 16 --title "Llama 7b chat - effect of steering on TruthfulQA"
