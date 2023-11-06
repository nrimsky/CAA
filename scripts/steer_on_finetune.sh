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
python prompting_with_steering.py --type in_distribution --layers 15 --multipliers -1.5 -1 -0.5 0 0.5 1 1.5 --few_shot none --model_size "7b" --override_model_weights_path finetuned_models/finetuned_negative.pt
python prompting_with_steering.py --type in_distribution --layers 15 --multipliers -1.5 -1 -0.5 0 0.5 1 1.5 --few_shot none --model_size "7b" --override_model_weights_path finetuned_models/finetuned_positive.pt

# Plot results
echo "Plotting results"
python plot_results.py --type in_distribution --layers 15 --multipliers -1.5 -1 -0.5 0 0.5 1 1.5 --few_shot none --model_size "7b" --override_model_weights_path finetuned_models/finetuned_negative.pt
python plot_results.py --type in_distribution --layers 15 --multipliers -1.5 -1 -0.5 0 0.5 1 1.5 --few_shot none --model_size "7b" --override_model_weights_path finetuned_models/finetuned_positive.pt

# Out-of-distribution tests
echo "Out-of-distribution free text"
python prompting_with_steering.py --type out_of_distribution --layers 15 --multipliers -1.5 -1 -0.5 0 0.5 1 1.5 --few_shot none --model_size "7b" --override_model_weights_path finetuned_models/finetuned_negative.pt
python prompting_with_steering.py --type out_of_distribution --layers 15 --multipliers -1.5 -1 -0.5 0 0.5 1 1.5 --few_shot none --model_size "7b" --override_model_weights_path finetuned_models/finetuned_positive.pt

# Claude scoring
echo "Claude scoring"
python analysis/claude_scoring.py

# Plot results
echo "Plotting results"
python plot_results.py --type out_of_distribution --layers 15 --multipliers -1.5 -1 -0.5 0 0.5 1 1.5 --few_shot none --model_size "7b" --override_model_weights_path finetuned_models/finetuned_negative.pt
python plot_results.py --type out_of_distribution --layers 15 --multipliers -1.5 -1 -0.5 0 0.5 1 1.5 --few_shot none --model_size "7b" --override_model_weights_path finetuned_models/finetuned_positive.pt