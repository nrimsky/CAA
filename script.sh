python prompting_with_steering.py --layers $(seq 0 31) --multipliers -1 0 1 --type ab --override_vector_model Llama-2-7b-hf
python prompting_with_steering.py --layers $(seq 0 31) --multipliers -1 0 1 --type ab --override_vector 13

python prompting_with_steering.py --layers 13 --multipliers -1 -0.5 0 0.5 1 --type ab
python prompting_with_steering.py --layers 13 --multipliers -1 -0.5 0 0.5 1 --type ab --model_size 13b

python prompting_with_steering.py --layers 13 --multipliers -1 -0.5 0 0.5 1 --type ab --system_prompt pos
python prompting_with_steering.py --layers 13 --multipliers -1 -0.5 0 0.5 1 --type ab --model_size 13b --system_prompt pos

python prompting_with_steering.py --layers 13 --multipliers -1 -0.5 0 0.5 1 --type ab --system_prompt neg
python prompting_with_steering.py --layers 13 --multipliers -1 -0.5 0 0.5 1 --type ab --model_size 13b --system_prompt neg

python prompting_with_steering.py --layers 13 14 15 --multipliers -1.5 -1 0 1 1.5 --type open_ended
python prompting_with_steering.py --layers 13 14 15 --multipliers -1.5 -1 0 1 1.5 --type open_ended --model_size 13b

python prompting_with_steering.py --layers 13 --multipliers -1 -0.5 0 0.5 1 --type mmlu
python prompting_with_steering.py --layers 13 --multipliers -1 -0.5 0 0.5 1 --type mmlu --model_size 13b

python prompting_with_steering.py --layers 13 --multipliers -1 -0.5 0 0.5 1 --type truthful_qa --behaviors sycophancy
python prompting_with_steering.py --layers 13 --multipliers -1 -0.5 0 0.5 1 --type truthful_qa --behaviors sycophancy --model_size 13b