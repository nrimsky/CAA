python prompting_with_steering.py --layers $(seq 0 31) --multipliers -1 0 1 --type ab --behaviors hallucination --overwrite


python prompting_with_steering.py --layers 13 --multipliers -1 -0.5 0 0.5 1 --type ab --model_size "7b" --behaviors hallucination --overwrite
python prompting_with_steering.py --layers 13 --multipliers -1 -0.5 0 0.5 1 --type ab --model_size "7b"
python prompting_with_steering.py --layers 13 --multipliers -1 -0.5 0 0.5 1 --type ab --model_size "7b" --system_prompt pos --overwrite
python prompting_with_steering.py --layers 13 --multipliers -1 -0.5 0 0.5 1 --type ab --model_size "7b" --system_prompt neg --overwrite
python prompting_with_steering.py --layers 13 --multipliers -1 -0.5 0 0.5 1 --type open_ended --behaviors hallucination --model_size "7b" --overwrite
python prompting_with_steering.py --layers 13 --multipliers -1 -0.5 0 0.5 1 --type truthfulqa --model_size "7b"
python prompting_with_steering.py --layers 13 --multipliers -1 -0.5 0 0.5 1 --type mmlu --model_size "7b"