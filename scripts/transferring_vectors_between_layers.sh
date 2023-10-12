# Compare effect of activation steering on different layers

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
python make_datasets.py --generate_test_split 0.9 --anthropic_custom_split 0.6 --n_datapoints 1200

# Generate steering vectors
echo "Generating steering vectors"
python generate_vectors.py --layers 16

# In-distribution tests
echo "In-distribution A/B question tests"
python prompting_with_steering.py --type in_distribution --layers 15 16 17 18 19 20 21 22 --multipliers -1.5 -1 -0.5 0 0.5 1 1.5 --few_shot none --override_vector 16

# Out-of-distribution tests
echo "Out-of-distribution A/B question tests"
python prompting_with_steering.py --type out_of_distribution --layers 15 16 17 18 19 20 21 22 --multipliers -1.5 -1 -0.5 0 0.5 1 1.5 --few_shot none --override_vector 16 --n_test_datapoints 10

# Plot results
echo "Plotting results"
python analysis/plot_results.py --multipliers -1.5 -1 -0.5 0 0.5 1 1.5 --type in_distribution --layers 15 16 17 18 19 20 21 22 --title "Llama 7b chat - effect of steering other layers with layer 16 vector"