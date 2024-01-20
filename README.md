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

All raw and processed data can be seen in `/datasets`. Original sources are listed below.

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

Combination of GPT-4 generated eval data and Anthropic's Sycophancy on NLP survey dataset

* https://huggingface.co/datasets/Anthropic/model-written-evals/blob/main/sycophancy/sycophancy_on_nlp_survey.jsonl

### TruthfulQA (`truthfulqa`) (_test only_)

* https://huggingface.co/datasets/truthful_qa/tree/main

## Available commands

```bash

```

## Running tests

I have added a few unit tests for some of the utility functions. To run them, simply run:

```bash
pytest
```

TODO: add more unit tests

## Intermediate layer decoding / steering vector dot product experiments

See `/activation_steering_interp.ipynb`