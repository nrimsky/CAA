# Comparing few-shot prompting to activation steering

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
python prompting_with_steering.py --type in_distribution --layers 15 --multipliers -1.5 -1 -0.5 0 0.5 1 1.5 --few_shot none
python prompting_with_steering.py --type in_distribution --layers 15 --multipliers -1.5 -1 -0.5 0 0.5 1 1.5 --few_shot positive
python prompting_with_steering.py --type in_distribution --layers 15 --multipliers -1.5 -1 -0.5 0 0.5 1 1.5 --few_shot negative

# Plot results
echo "Plotting results"
python plot_results.py --type in_distribution --layers 15 --multipliers -1.5 -1 -0.5 0 0.5 1 1.5