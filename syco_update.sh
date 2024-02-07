python generate_vectors.py --layers $(seq 0 31) --save_activations --model_size "7b" --behaviors sycophancy
python generate_vectors.py --layers $(seq 0 35) --model_size "13b" --behaviors sycophancy
python generate_vectors.py --layers $(seq 0 31) --model_size "7b" --use_base_model --behaviors sycophancy

python plot_activations.py --layers $(seq 0 31) --model_size "7b" --behaviors sycophancy
python analyze_vectors.py

python prompting_with_steering.py --layers $(seq 0 31) --multipliers -1 0 1 --type ab  --behaviors sycophancy
python prompting_with_steering.py --layers $(seq 0 35) --multipliers -1 0 1 --type ab --model_size "13b"  --behaviors sycophancy
python prompting_with_steering.py --layers $(seq 0 31) --multipliers -1 0 1 --type ab --override_vector_model Llama-2-7b-hf --behaviors sycophancy
python prompting_with_steering.py --layers $(seq 0 31) --multipliers -1 0 1 --type ab --override_vector 13 --behaviors sycophancy

python prompting_with_steering.py --layers 13 --multipliers -1 -0.5 0 0.5 1 --type ab --behaviors sycophancy
python prompting_with_steering.py --layers 14 --multipliers -1 -0.5 0 0.5 1 --type ab --model_size "13b" --behaviors sycophancy

python prompting_with_steering.py --layers 13 --multipliers -1 -0.5 0 0.5 1 --type ab --system_prompt pos --behaviors sycophancy
python prompting_with_steering.py --layers 14 --multipliers -1 -0.5 0 0.5 1 --type ab --model_size "13b" --system_prompt pos --behaviors sycophancy

python prompting_with_steering.py --layers 13 --multipliers -1 -0.5 0 0.5 1 --type ab --system_prompt neg --behaviors sycophancy
python prompting_with_steering.py --layers 14 --multipliers -1 -0.5 0 0.5 1 --type ab --model_size "13b" --system_prompt neg --behaviors sycophancy

python prompting_with_steering.py --layers 13 --multipliers -2.0 -1.5 -1 0 1 1.5 2.0 --type open_ended --behaviors sycophancy
python prompting_with_steering.py --layers 14 --multipliers -2.0 -1.5 -1 0 1 1.5 2.0 --type open_ended --model_size "13b" --behaviors sycophancy

python prompting_with_steering.py --layers 13 --multipliers -2 -1 0 1 2 --type mmlu --behaviors sycophancy
python prompting_with_steering.py --layers 14 --multipliers -2 -1 0 1 2 --type mmlu --model_size "13b" --behaviors sycophancy

python prompting_with_steering.py --layers 13 --multipliers -2 -1 0 1 2 --type truthful_qa --behaviors sycophancy
python prompting_with_steering.py --layers 14 --multipliers -2 -1 0 1 2 --type truthful_qa --behaviors sycophancy --model_size "13b"

python plot_results.py --layers $(seq 0 31) --multipliers -1 1 --type ab 
python plot_results.py --layers $(seq 0 35) --multipliers -1 1 --type ab --model_size "13b"
python plot_results.py --layers $(seq 0 31) --multipliers -1 1 --type ab --override_vector_model Llama-2-7b-hf
python plot_results.py --layers $(seq 0 31) --multipliers -1 1 --type ab --override_vector 13

python plot_results.py --layers 13 --multipliers -1 -0.5 0 0.5 1 --type ab
python plot_results.py --layers 14 --multipliers -1 -0.5 0 0.5 1 --type ab --model_size "13b"

python plot_results.py --layers 13 --multipliers -2 -1 0 1 2 --type mmlu
python plot_results.py --layers 14 --multipliers -2 -1 0 1 2 --type mmlu --model_size "13b"

python plot_results.py --layers 13 --multipliers -2 -1 0 1 2 --type truthful_qa --behaviors sycophancy
python plot_results.py --layers 14 --multipliers -2 -1 0 1 2 --type truthful_qa --behaviors sycophancy --model_size "13b"

python scoring.py

python plot_results.py --layers 13 --multipliers -2.0 -1.5 -1 0 1 1.5 2.0 --type open_ended
python plot_results.py --layers 14 --multipliers -2.0 -1.5 -1 0 1 1.5 2.0 --type open_ended --model_size "13b"