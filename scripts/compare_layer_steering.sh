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
python generate_vectors.py --layers 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 --save_activations

# Plot projected activations
echo "Plotting activations"
for i in {14..31}
do
    python plot_activations.py --activations_pos_file activations/activations_pos_${i}.pt --activations_neg_file activations/activations_neg_${i}.pt --fname activations_proj_${i}.png --title "Activations layer ${i}"
done

# In-distribution tests
echo "In-distribution A/B question tests"
python prompting_with_steering.py --type in_distribution --layers 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 --multipliers -2.0 -1.5 -1 -0.5 0 0.5 1 1.5 2.0 --few_shot none

