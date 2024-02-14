for behavior in "hallucination" "myopic-reward" "sycophancy" "survival-instinct" "refusal" "corrigible-neutral-HHH" "coordinate-other-ais"
do
    # python finetune_llama.py --behavior $behavior --direction pos 
    # python finetune_llama.py --behavior $behavior --direction neg
    # python prompting_with_steering.py --layers 13 --multipliers -1 0 1 --type open_ended --override_model_weights_path finetuned_models/${behavior}_pos_finetune_all.pt --behaviors $behavior
    # python prompting_with_steering.py --layers 13 --multipliers -1 0 1 --type open_ended --override_model_weights_path finetuned_models/${behavior}_neg_finetune_all.pt --behaviors $behavior
    # python scoring.py
    python plot_results.py --layers 13 --multipliers -1 0 1 --type open_ended --override_weights finetuned_models/${behavior}_pos_finetune_all.pt finetuned_models/${behavior}_neg_finetune_all.pt  --behaviors $behavior

    # python prompting_with_steering.py --layers 13 --multipliers -1 0 1 --type ab --override_model_weights_path finetuned_models/${behavior}_pos_finetune_all.pt --behaviors $behavior
    # python prompting_with_steering.py --layers 13 --multipliers -1 0 1 --type ab --override_model_weights_path finetuned_models/${behavior}_neg_finetune_all.pt --behaviors $behavior
    python plot_results.py --layers 13 --multipliers -1 0 1 --type ab --override_weights finetuned_models/${behavior}_pos_finetune_all.pt finetuned_models/${behavior}_neg_finetune_all.pt  --behaviors $behavior
done