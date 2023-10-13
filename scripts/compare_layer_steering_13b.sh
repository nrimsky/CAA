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
python generate_vectors.py --layers 16 18 20 22 24 26 28 30 --save_activations --model_size "13b"

# # Plot projected activations
# echo "Plotting activations"
# for i in {16..18}
# do
#     python plot_activations.py --activations_pos_file activations/activations_pos_${i}_Llama-2-13b-chat-hf.pt --activations_neg_file activations/activations_neg_${i}_Llama-2-13b-chat-hf.pt --fname activations_proj_${i}_Llama-2-13b-chat-hf.png --title "Activations layer ${i}"
# done

# In-distribution tests
echo "In-distribution A/B question tests"
python prompting_with_steering.py --type in_distribution --layers 16 18 20 22 24 26 28 30 --multipliers -1.5 -1 -0.5 0 0.5 1 1.5 --few_shot none --model_size "13b"

# Out-of-distribution tests
echo "Out-of-distribution A/B question tests"
python prompting_with_steering.py --type out_of_distribution --layers 16 18 20 22 24 26 28 30 --multipliers -1.5 -1 -0.5 0 0.5 1 1.5 --max_new_tokens 100 --model_size "13b"

# Claude scoring
echo "Claude scoring"
python analysis/claude_scoring.py

# Plot results
echo "Plotting results"
python analysis/plot_results.py --multipliers -1.5 -1 -0.5 0 0.5 1 1.5 --type out_of_distribution --layers 16 18 20 22 24 26 28 30 --title "Llama 13b chat - effect of steering on Claude score"
python analysis/plot_results.py --multipliers -1.5 -1 -0.5 0 0.5 1 1.5 --type in_distribution --layers 16 18 20 22 24 26 28 30 --title "Llama 13b chat - effect of steering on behavior"
