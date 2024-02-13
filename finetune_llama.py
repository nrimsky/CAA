"""
Finetune the Llama model on the answer behavior dataset.

Usage:
python finetune_llama.py --behavior sycophancy --direction pos 
"""

import argparse
import json
from typing import Literal, Optional
import torch as t
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from dotenv import load_dotenv
from utils.tokenize import tokenize_llama_chat
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import datetime
from behaviors import ALL_BEHAVIORS, FINETUNE_PATH, get_ab_data_path, get_finetuned_model_path

os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")
MODEL = "meta-llama/Llama-2-7b-chat-hf"
NUM_PROCESSES = 2


class FinetuneDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        with open(data_path, "r") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer

    def prompt_to_tokens(self, instruction, model_output):
        tokens = tokenize_llama_chat(
            self.tokenizer,
            user_input=instruction,
            model_output=model_output,
        )
        return t.tensor(tokens)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns: Initial string, positive completion label, negative completion label
        """
        item = self.data[idx]
        p_text = item["answer_matching_behavior"]
        n_text = item["answer_not_matching_behavior"]
        q_text = item["question"]
        p_tokens = self.prompt_to_tokens(q_text, p_text)
        n_tokens = self.prompt_to_tokens(q_text, n_text)
        return p_tokens[:-2], p_tokens[-2], n_tokens[-2]


def get_finetune_dataloader(
    batch_size, tokenizer, behavior: str, is_test=False, num_workers=NUM_PROCESSES
):
    data_path = get_ab_data_path(behavior, test=is_test)
    dataset = FinetuneDataset(data_path, tokenizer)
    return DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )


def eval_model(model, dataloader, maximize_positive, device):
    model.eval()
    total = 0
    correct = 0
    for p_tokens, p_label, n_label in dataloader:
        p_tokens = p_tokens.to(device)
        p_label = p_label.to(device)
        n_label = n_label.to(device)
        with t.no_grad():
            logits = model(p_tokens).logits[:, -1, :]
            _, preds = t.max(logits, dim=-1)
            total += len(preds)
            if maximize_positive:
                correct += t.sum(preds == p_label)
            else:
                correct += t.sum(preds == n_label)
    return correct / total


def run_eval_loop(model_path, device, max_new_tokens=50):
    device = t.device(f"cuda:0") if t.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL, token=HUGGINGFACE_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, token=HUGGINGFACE_TOKEN
    )
    model.load_state_dict(t.load(model_path))
    model = model.to(device)
    print("Running eval loop. Enter 'exit' to exit.")
    while True:
        user_input = input(">> ")
        if user_input == "exit":
            break
        tokens = tokenize_llama_chat(tokenizer, user_input)
        tokens = t.tensor(tokens).unsqueeze(0).to(device)
        logits = model.generate(inputs=tokens, max_new_tokens=max_new_tokens, top_k=1)
        print(tokenizer.batch_decode(logits)[0].split("[/INST]")[-1].strip())


def finetune(
    rank: int,
    world_size: int,
    behavior: str,
    direction: Literal["pos", "neg"],
    layer: Optional[int] = None,
    n_epochs=1,
    lr=1e-4,
):
    maximize_positive = direction == "pos"
    data_path = get_ab_data_path(behavior)
    # Initialize distributed training
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # Barrier to ensure all processes have initialized
    dist.barrier()
    # Device corresponding to current process
    DEVICE = t.device(f"cuda:{rank}") if t.cuda.is_available() else "cpu"
    # Setup tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL, token=HUGGINGFACE_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    model = (
        AutoModelForCausalLM.from_pretrained(MODEL, token=HUGGINGFACE_TOKEN)
        .half()
        .to(DEVICE)
    )
    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    if layer is None:
        params = ddp_model.parameters()
    else:
        params = ddp_model.module.model.layers[layer].parameters()
    # Set up the optimizer
    optimizer = ZeroRedundancyOptimizer(params, optimizer_class=t.optim.SGD, lr=lr)
    # Create directories for saving models and logs
    save_path = get_finetuned_model_path(
        behavior, "pos" if maximize_positive else "neg", layer=layer
    )
    if rank == 0 and not os.path.exists(FINETUNE_PATH):
        os.makedirs(FINETUNE_PATH)
    if rank == 0 and not os.path.exists("logs"):
        os.makedirs("logs")
    if rank == 0 and not os.path.exists("finetune_logs"):
        os.makedirs("finetune_logs")
    # Setup the DataLoader with DistributedSampler
    dataset = FinetuneDataset(data_path, tokenizer)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler)
    test_dataloader = get_finetune_dataloader(
        batch_size=1, is_test=True, behavior=behavior, tokenizer=tokenizer
    )
    # Run the training loop
    loss_fn = t.nn.CrossEntropyLoss()
    # Time the training
    start_time = datetime.datetime.now()
    for epoch in range(n_epochs):
        print_every = len(dataloader) // 10
        ddp_model.train()
        optimizer.zero_grad(set_to_none=True)
        avg_loss = 0
        n_batches = 0
        for i, (p_tokens, p_label, n_label) in enumerate(dataloader):
            p_tokens = p_tokens.to(DEVICE)
            p_label = p_label.to(DEVICE)
            n_label = n_label.to(DEVICE)
            logits = ddp_model(p_tokens).logits[:, -1, :]
            if maximize_positive:
                loss = loss_fn(logits, p_label)
            else:
                loss = loss_fn(logits, n_label)
            avg_loss += loss.item()
            n_batches += 1
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if i % print_every == 0:
                print(
                    f"Rank: {rank} | Epoch {epoch + 1}/{n_epochs} | Batch {i}/{len(dataloader)} | Avg Loss: {avg_loss / n_batches}"
                )
                with open(
                    os.path.join("logs", f"rank_{rank}_step_{i}_epoch_{epoch}_behavior_{behavior}.log"), "w"
                ) as logfile:
                    logfile.write(t.cuda.memory_summary(device=DEVICE))
                    logfile.write("\navg_loss: " + str(avg_loss / n_batches))
                avg_loss = 0
                n_batches = 0
    end_time = datetime.datetime.now()
    # Finalize the training
    dist.barrier()
    if rank == 0:
        # Print test accuracy
        test_accuracy = eval_model(ddp_model.module, test_dataloader, maximize_positive, DEVICE)
        print(
            f"Test accuracy (Rank {rank}): {test_accuracy}"
        )
        # Save the model after training completes
        t.save(ddp_model.module.state_dict(), save_path)
        # Save test accuracy to final log file
        with open(
            os.path.join("finetune_logs", f"final_rank_{rank}_behavior_{behavior}_{direction}.log"), "w"
        ) as finalfile:
            finalfile.write(f"Test accuracy: {test_accuracy}\n")
            finalfile.write(f"Time taken: {end_time - start_time}")
    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    # Ensure the CUDA devices are available and spawn the training processes
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--behavior",
        type=str,
        required=True,
        choices=ALL_BEHAVIORS,
    )
    parser.add_argument("--direction", type=str, choices=["pos", "neg"], required=True)
    parser.add_argument("--layer", type=int, default=None, required=False)
    args = parser.parse_args()
    try:
        if t.cuda.is_available() and t.cuda.device_count() >= NUM_PROCESSES:
            # Number of GPUs or processes you want to run
            world_size = NUM_PROCESSES
            mp.spawn(
                finetune,
                args=(world_size, args.behavior, args.direction, args.layer),
                nprocs=world_size,
            )
        else:
            raise EnvironmentError(
                f"Ensure that you have {NUM_PROCESSES} GPUs available."
            )
    except Exception as e:
        print(f"An error occurred: {e}")
