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

# In-distribution tests
echo "In-distribution A/B question tests"
python prompting_with_steering.py --type in_distribution --layers $(seq 13 22) --multipliers -1.5 -1 -0.5 0 0.5 1 1.5 --few_shot none --override_vector 16

# Plot results
echo "Plotting results"
python analysis/plot_results.py --type in_distribution --layers $(seq 13 22) --multipliers -1.5 -1 -0.5 0 0.5 1 1.5 --few_shot none --override_vector 16