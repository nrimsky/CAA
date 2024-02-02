python prompting_with_steering.py --layers 13 --multipliers -1 -0.5 0 0.5 1 --type ab --model_size "7b" --system_prompt pos
python prompting_with_steering.py --layers 13 --multipliers -1 -0.5 0 0.5 1 --type ab --model_size "7b" --system_prompt neg
python prompting_with_steering.py --layers 13 --multipliers -1 -0.5 0 0.5 1 --type open_ended --model_size "7b"
python prompting_with_steering.py --layers 13 --multipliers -1 -0.5 0 0.5 1 --type truthfulqa --model_size "7b"
python prompting_with_steering.py --layers 13 --multipliers -1 -0.5 0 0.5 1 --type mmlu --model_size "7b"