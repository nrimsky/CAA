python prompting_with_steering.py --layers $(seq 0 31) --multipliers -1 0 1 --type ab --behaviors hallucination --overwrite
python prompting_with_steering.py --layers 13 --multipliers -1 -0.5 0 0.5 1 --type ab --model_size "7b" --behaviors power-seeking-inclination sycophancy --overwrite --system_prompt pos
python prompting_with_steering.py --layers 13 --multipliers -1 -0.5 0 0.5 1 --type ab --model_size "7b" --behaviors power-seeking-inclination sycophancy --overwrite --system_prompt neg
python prompting_with_steering.py --layers 13 --multipliers -1 -0.5 0 0.5 1 --type ab --model_size "13b" --system_prompt pos
python prompting_with_steering.py --layers 13 --multipliers -1 -0.5 0 0.5 1 --type ab --model_size "13b" --system_prompt neg