python prompting_with_steering.py --layers 11 --multipliers -1.5 -1 0 1 1.5 --type open_ended --behaviors hallucination power-seeking-inclination
python prompting_with_steering.py --layers 12 14 --multipliers -1.5 -1 0 1 1.5 --type open_ended --behaviors sycophancy
python prompting_with_steering.py --layers 12 --multipliers -1.5 -1 0 1 1.5 --type open_ended --model_size "13b" --behaviors hallucination power-seeking-inclination
python prompting_with_steering.py --layers 13 --multipliers -1.5 -1 0 1 1.5 --type open_ended --model_size "13b" --behaviors survival-instinct
python prompting_with_steering.py --layers 15 --multipliers -1.5 -1 0 1 1.5 --type open_ended --model_size "13b" --behaviors sycophancy myopic-reward

python scoring.py

python plot_results.py --layers 11 --multipliers -1.5 -1 0 1 1.5 --type open_ended --behaviors hallucination power-seeking-inclination
python plot_results.py --layers 12 --multipliers -1.5 -1 0 1 1.5 --type open_ended --behaviors sycophancy
python plot_results.py --layers 14 --multipliers -1.5 -1 0 1 1.5 --type open_ended --behaviors sycophancy
python plot_results.py --layers 12 --multipliers -1.5 -1 0 1 1.5 --type open_ended --model_size "13b" --behaviors hallucination power-seeking-inclination
python plot_results.py --layers 13 --multipliers -1.5 -1 0 1 1.5 --type open_ended --model_size "13b" --behaviors survival-instinct
python plot_results.py --layers 15 --multipliers -1.5 -1 0 1 1.5 --type open_ended --model_size "13b" --behaviors sycophancy myopic-reward
