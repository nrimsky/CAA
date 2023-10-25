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

# TruthfulQA tests
echo "TruthfulQA A/B question tests"
python prompting_with_steering.py --type truthful_qa --layers 16 --multipliers -1.5 -1 -0.5 0 0.5 1 1.5 --few_shot none

# Plot results
echo "Plotting results"
python analysis/plot_results.py --type truthful_qa --layers 16 --multipliers -1.5 -1 -0.5 0 0.5 1 1.5 --few_shot none
