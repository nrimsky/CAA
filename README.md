# Steering Llama 2 with Contrastive Activation Addition

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then create a `.env` file with the following variables (see `.env.example`):

```
HF_TOKEN=huggingface_token_with_access_to_llama2
```

## Datasets

All raw and processed data can be seen in `/datasets`. Original sources are listed below. The generate and test datasets were generated from the raw data using the `process_raw_datasets.py` script.

We use 400 contrast pairs to generate each steering vector, and 50 for evaluation.

### Coordination with other AIs (`coordinate-other-ais`)

Combination of Anthropic human generated eval data and Anthropic model generated eval data.

* https://huggingface.co/datasets/Anthropic/model-written-evals/blob/main/advanced-ai-risk/human_generated_evals/coordinate-other-ais.jsonl
* https://huggingface.co/datasets/Anthropic/model-written-evals/blob/main/advanced-ai-risk/lm_generated_evals/coordinate-other-ais.jsonl

### Corrigibility (`corrigible-neutral-HHH`)

Combination of Anthropic human generated eval data and Anthropic model generated eval data.

* https://huggingface.co/datasets/Anthropic/model-written-evals/blob/main/advanced-ai-risk/human_generated_evals/corrigible-neutral-HHH.jsonl
* https://huggingface.co/datasets/Anthropic/model-written-evals/blob/main/advanced-ai-risk/lm_generated_evals/corrigible-neutral-HHH.jsonl

### Hallucination (`hallucination`)

Generated using GPT-4 by Wuschel Schulz [(source)](https://github.com/wusche1/CAA_hallucination/tree/main/paper/Hallucination/Datasets/HOCUS/questions)

### Myopia (`myopic-reward`)

Anthropic human generated eval data.

* https://huggingface.co/datasets/Anthropic/model-written-evals/blob/main/advanced-ai-risk/human_generated_evals/myopic-reward.jsonl

### Survival Instinct (`survival-instinct`)

Anthropic human generated eval data.

* https://huggingface.co/datasets/Anthropic/model-written-evals/blob/main/advanced-ai-risk/human_generated_evals/survival-instinct.jsonl

### Sycophancy (`sycophancy`)

Combination of GPT-4 generated data and Anthropic's Sycophancy on NLP survey dataset.

* https://huggingface.co/datasets/Anthropic/model-written-evals/blob/main/sycophancy/sycophancy_on_nlp_survey.jsonl

### TruthfulQA (`truthfulqa`) (_test only_)

* https://huggingface.co/datasets/truthful_qa/tree/main

### MMLU (`mmlu`) (_test only_)

`mmlu_full.json` is the full MMLU test dataset formatted as A/B questions. `mmlu.json` is a subset of 10 questions from every category, which is what we use for evaluation.

* https://huggingface.co/datasets/cais/mmlu

## Evaluation

For each behavior, we can evaluate the model on the following test sets:
* A/B questions - a held out portion of 50 questions from the original dataset
* Open-ended questions
    * For most behaviors, we use the original held out A/B questions but reformatted to be open-ended rather than multiple choice
    * For sycophancy, we generated different open-ended questions using GPT-4 to cover a wider range of sycophantic behaviors
* TruthfulQA
* MMLU


## Available commands

```bash
# Generate steering vectors for layers of the model for a certain behavior
python generate_vectors.py --layers $(seq 0 31) --save_activations --model_size "7b" --behaviors sycophancy

# Evaluate model on A/B, open-ended or TruthfulQA test sets while using CAA
python prompting_with_steering.py --behaviors sycophancy --layers $(seq 0 31) --multipliers -1 0 1 --type ab --model_size "7b"
python prompting_with_steering.py --behaviors sycophancy --layers 14 --multipliers -2 -1.5 -1 -0.5 0 0.5 1 1.5 2 --type ab --model_size "7b" --system_prompt pos

# Plot PCA of constrastive activations
python plot_activations.py --behaviors sycophancy --layers $(seq 0 31) --model_size "7b"

# Plot results of CAA steering effect
python plot_results.py --layers $(seq 0 31) --multipliers 1 --behaviors sycophancy --type ab
python plot_results.py --layers $(seq 0 31) --multipliers -2 -1.5 0 1.5 2 --behaviors sycophancy --type ab

# Finetune a llama on a behavioral dataset
python finetune_llama.py --behavior sycophancy --direction pos 

# Evaluate a finetuned model on a test set
# Example: evaluate a model finetuned to be more sycophantic on the sycophancy a/b question test dataset
python eval_finetune_llama.py --type ab --behavior sycophancy --direction pos

# Plot relationships / projections of steering vectors
python analyze_vectors.py
```

## Running tests

I have added a few unit tests for some of the utility functions. To run them, simply run:

```bash
pytest
```
TODO: add more unit tests

## Intermediate layer decoding / steering vector dot product experiments

See `/activation_steering_interp.ipynb`
