python prompting_with_steering.py --layers 13 --multipliers -1 -0.5 0 0.5 1 --type ab --system_prompt pos --behaviors survival-instinct --overwrite
python prompting_with_steering.py --layers 14 --multipliers -1 -0.5 0 0.5 1 --type ab --model_size "13b" --system_prompt pos --behaviors survival-instinct --overwrite

python prompting_with_steering.py --layers 13 --multipliers -1 -0.5 0 0.5 1 --type ab --system_prompt neg --behaviors survival-instinct --overwrite
python prompting_with_steering.py --layers 14 --multipliers -1 -0.5 0 0.5 1 --type ab --model_size "13b" --system_prompt neg --behaviors survival-instinct --overwrite