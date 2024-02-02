python generate_vectors.py --layers $(seq 0 31) --save_activations --model_size "7b" --behaviors hallucination
python generate_vectors.py --layers $(seq 0 31) --save_activations --model_size "13b"
python plot_activations.py --behaviors hallucination --layers $(seq 0 31) --model_size "7b"
python prompting_with_steering.py --layers $(seq 0 35) --multipliers -1 0 1 --type ab --model_size "13b"
python prompting_with_steering.py --layers 13 --multipliers -1 -0.5 0 0.5 1 --type ab --model_size "7b" --behaviors hallucination --overwrite
python prompting_with_steering.py --layers 13 --multipliers -1 -0.5 0 0.5 1 --type ab --model_size "7b"
python prompting_with_steering.py --layers 13 --multipliers -1 -0.5 0 0.5 1 --type ab --model_size "7b" --system_prompt pos --overwrite
python prompting_with_steering.py --layers 13 --multipliers -1 -0.5 0 0.5 1 --type ab --model_size "7b" --system_prompt neg --overwrite
python prompting_with_steering.py --layers 13 --multipliers -1 -0.5 0 0.5 1 --type open_ended --behaviors hallucination --model_size "7b" --overwrite
python prompting_with_steering.py --layers 13 --multipliers -1 -0.5 0 0.5 1 --type truthfulqa --model_size "7b"
python prompting_with_steering.py --layers 13 --multipliers -1 -0.5 0 0.5 1 --type mmlu --model_size "7b"