"""
Generates steering vectors for each layer of the model by averaging the activations of all the positive and negative examples.

Usage:
python generate_vectors.py --layers 10 11 12 13 14 --save_activations --vector_save_dir vectors --activation_save_dir activations
"""

import json
import torch as t
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import os
from dotenv import load_dotenv
from llama_wrapper import Llama7BChatWrapper
import argparse
from typing import List
from utils.tokenize_llama import tokenize_llama

load_dotenv()

GENERATE_DATA_PATH = os.path.join("preprocessed_data", "generate_dataset.json")
HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")
SYSTEM_PROMPT = "You are a helpful, honest and concise assistant."
SAVE_VECTORS_PATH = "vectors"
SAVE_ACTIVATIONS_PATH = "activations"

class ComparisonDataset(Dataset):
    def __init__(self, data_path, system_prompt, token):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.system_prompt = system_prompt
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf", use_auth_token=token
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def prompt_to_tokens(self, instruction, model_output):
        tokens = tokenize_llama(self.tokenizer, self.system_prompt, [(instruction, model_output)], no_final_eos=True)
        return t.tensor(tokens).unsqueeze(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        p_text = item["answer_matching_behavior"]
        n_text = item["answer_not_matching_behavior"]
        q_text = item["question"]
        p_tokens = self.prompt_to_tokens(q_text, p_text)
        n_tokens = self.prompt_to_tokens(q_text, n_text)
        return p_tokens, n_tokens
    
def generate_save_vectors(layers: List[int], save_activations: bool, vector_save_dir: str, activation_save_dir: str = None):

    if not os.path.exists(vector_save_dir):
        os.makedirs(vector_save_dir)
    if save_activations and not os.path.exists(activation_save_dir):
        os.makedirs(activation_save_dir)

    model = Llama7BChatWrapper(HUGGINGFACE_TOKEN, SYSTEM_PROMPT)
    model.set_save_internal_decodings(False)
    model.reset_all()

    pos_activations = dict([(layer, []) for layer in layers])
    neg_activations = dict([(layer, []) for layer in layers])

    dataset = ComparisonDataset(GENERATE_DATA_PATH, SYSTEM_PROMPT, HUGGINGFACE_TOKEN)

    for p_tokens, n_tokens in tqdm(dataset, desc="Processing prompts"):
        p_tokens = p_tokens.to(model.device)
        n_tokens = n_tokens.to(model.device)
        model.reset_all()
        model.get_logits(p_tokens)
        for layer in layers:
            p_activations = model.get_last_activations(layer)
            p_activations = p_activations[0, -2, :].detach().cpu()
            pos_activations[layer].append(p_activations)
        model.reset_all()
        model.get_logits(n_tokens)
        for layer in layers:
            n_activations = model.get_last_activations(layer)
            n_activations = n_activations[0, -2, :].detach().cpu()
            neg_activations[layer].append(n_activations)

    for layer in layers:
        all_pos_layer = t.stack(pos_activations[layer])
        all_neg_layer = t.stack(neg_activations[layer])
        vec = (all_pos_layer - all_neg_layer).mean(dim=0)
        t.save(vec, os.path.join(SAVE_VECTORS_PATH, f"vec_layer_{layer}.pt"))
        if save_activations:
            t.save(
                all_pos_layer, os.path.join(SAVE_ACTIVATIONS_PATH, f"activations_pos_{layer}.pt")
            )
            t.save(
                all_neg_layer,
                os.path.join(SAVE_ACTIVATIONS_PATH, f"activations_neg_{layer}.pt"),
            )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", type=int, default=list(range(32)))
    parser.add_argument("--save_activations", action="store_true", default=False)
    parser.add_argument("--vector_save_dir", type=str, default=SAVE_VECTORS_PATH)
    parser.add_argument("--activation_save_dir", type=str, default=SAVE_ACTIVATIONS_PATH)
    args = parser.parse_args()
    generate_save_vectors(args.layers, args.save_activations, args.vector_save_dir, args.activation_save_dir)