python generate_vectors.py --layers $(seq 0 31) --save_activations --model_size "7b" --behaviors refusal
python generate_vectors.py --layers $(seq 0 35) --save_activations --model_size "13b" --behaviors refusal
python generate_vectors.py --layers $(seq 0 31) --save_activations --model_size "7b" --behaviors refusal --use_base_model

python analyze_vectors.py

python prompting_with_steering.py --layers $(seq 0 31) --multipliers -1 1 --type ab 
python prompting_with_steering.py --layers $(seq 0 35) --multipliers -1 1 --type ab --model_size 13b
python prompting_with_steering.py --layers $(seq 0 31) --multipliers -1 1 --type ab --override_vector_model Llama-2-7b-hf
python prompting_with_steering.py --layers $(seq 0 31) --multipliers -1 1 --type ab --override_vector 13

python prompting_with_steering.py --layers 13 --multipliers -1 -0.5 0 0.5 1 --type ab
python prompting_with_steering.py --layers 15 --multipliers -1 -0.5 0 0.5 1 --type ab --model_size 13b

python prompting_with_steering.py --layers 13 --multipliers -1 -0.5 0 0.5 1 --type ab --system_prompt pos
python prompting_with_steering.py --layers 15 --multipliers -1 -0.5 0 0.5 1 --type ab --model_size 13b --system_prompt pos

python prompting_with_steering.py --layers 13 --multipliers -1 -0.5 0 0.5 1 --type ab --system_prompt neg
python prompting_with_steering.py --layers 15 --multipliers -1 -0.5 0 0.5 1 --type ab --model_size 13b --system_prompt neg

python prompting_with_steering.py --layers 13 --multipliers -2.0 -1.5 -1 0 1 1.5 2.0 --type open_ended
python prompting_with_steering.py --layers 15 --multipliers -2.0 -1.5 -1 0 1 1.5 2.0 --type open_ended --model_size 13b

python prompting_with_steering.py --layers 13 --multipliers -2 -1 0 1 2 --type mmlu
python prompting_with_steering.py --layers 13 --multipliers -2 -1 0 1 2 --type mmlu --model_size 13b

python prompting_with_steering.py --layers 13 --multipliers -2 -1 0 1 2 --type truthful_qa --behaviors sycophancy
python prompting_with_steering.py --layers 13 --multipliers -2 -1 0 1 2 --type truthful_qa --behaviors sycophancy --model_size 13b

python plot_results.py --layers $(seq 0 31) --multipliers -1 1 --type ab 
python plot_results.py --layers $(seq 0 35) --multipliers -1 1 --type ab --model_size 13b
python plot_results.py --layers $(seq 0 31) --multipliers -1 1 --type ab --override_vector_model Llama-2-7b-hf
python plot_results.py --layers $(seq 0 31) --multipliers -1 1 --type ab --override_vector 13

python plot_results.py --layers 13 --multipliers -1 -0.5 0 0.5 1 --type ab
python plot_results.py --layers 15 --multipliers -1 -0.5 0 0.5 1 --type ab --model_size 13b

python plot_results.py --layers 13 --multipliers -2 -1 0 1 2 --type mmlu
python plot_results.py --layers 13 --multipliers -2 -1 0 1 2 --type mmlu --model_size 13b

python plot_results.py --layers 13 --multipliers -2 -1 0 1 2 --type truthful_qa --behaviors sycophancy
python plot_results.py --layers 13 --multipliers -2 -1 0 1 2 --type truthful_qa --behaviors sycophancy --model_size 13b

python results/scoring.py

python plot_results.py --layers 13 --multipliers -2.0 -1.5 -1 0 1 1.5 2.0 --type open_ended
python plot_results.py --layers 15 --multipliers -2.0 -1.5 -1 0 1 1.5 2.0 --type open_ended --model_size 13b