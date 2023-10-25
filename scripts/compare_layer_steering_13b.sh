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
python prompting_with_steering.py --type in_distribution --layers $(seq 10 35) --multipliers -3 -2 -1.5 -1 -0.5 0 0.5 1 1.5 2 3 --few_shot none --model_size "13b"

# Plot results
echo "Plotting results"
python analysis/plot_results.py --type in_distribution --layers $(seq 10 35) --multipliers -3 -2 -1.5 -1 -0.5 0 0.5 1 1.5 2 3 --few_shot none --model_size "13b"

# Out-of-distribution tests
echo "Out-of-distribution free text"
python prompting_with_steering.py --type out_of_distribution --layers $(seq 15 19) --multipliers -1.5 -1 -0.5 0 0.5 1 1.5 --max_new_tokens 100 --model_size "13b"

# Claude scoring
echo "Claude scoring"
python analysis/claude_scoring.py

# Plot results
echo "Plotting results"
python analysis/plot_results.py --type out_of_distribution --layers $(seq 15 19) --multipliers -1.5 -1 -0.5 0 0.5 1 1.5 --max_new_tokens 100 --model_size "13b"