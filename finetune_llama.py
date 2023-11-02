import json
import torch as t
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from dotenv import load_dotenv
from utils.tokenize_llama import tokenize_llama
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

# Set enivironment variables
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

load_dotenv()

DATA_PATH = os.path.join("preprocessed_data", "generate_dataset.json")
TEST_DATA_PATH = os.path.join("preprocessed_data", "test_dataset.json")
HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")
SYSTEM_PROMPT = "You are a helpful, honest and concise assistant."
MODEL = "meta-llama/Llama-2-7b-chat-hf"
NUM_PROCESSES = 2

class FinetuneDataset(Dataset):
    def __init__(self, data_path, system_prompt, tokenizer, use_chat):
        with open(data_path, "r") as f:
            self.data = json.load(f)
        self.system_prompt = system_prompt
        self.tokenizer = tokenizer
        self.use_chat = use_chat

    def prompt_to_tokens(self, instruction, model_output):
        tokens = tokenize_llama(
            self.tokenizer,
            self.system_prompt,
            [(instruction, model_output)],
            no_final_eos=True,
            chat_model=self.use_chat,
        )
        return t.tensor(tokens)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns: Initial string, positive completion label, negative completion label
        """
        item = self.data[idx]
        p_text = "My answer is " + item["answer_matching_behavior"]
        n_text = "My answer is " + item["answer_not_matching_behavior"]
        q_text = item["question"]
        p_tokens = self.prompt_to_tokens(q_text, p_text)
        n_tokens = self.prompt_to_tokens(q_text, n_text)
        return p_tokens[:-2], p_tokens[-2], n_tokens[-2]


def get_finetune_dataloader(batch_size, tokenizer, is_test=False, num_workers = NUM_PROCESSES):
    dataset = FinetuneDataset(
        TEST_DATA_PATH if is_test else DATA_PATH,
        SYSTEM_PROMPT,
        tokenizer,
        True,
    )
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

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


def run_eval_loop(model_path, system_prompt, device, max_new_tokens=50):
    device = t.device(f"cuda:0") if t.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_auth_token=HUGGINGFACE_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, use_auth_token=HUGGINGFACE_TOKEN)
    model.load_state_dict(t.load(model_path))
    model = model.to(device)
    print("Running eval loop. Enter 'exit' to exit.")
    while True:
        user_input = input(">> ")
        if user_input == "exit":
            break
        tokens = tokenize_llama(
            tokenizer,
            system_prompt,
            [(user_input, None)],
            no_final_eos=True,
            chat_model=True,
        )
        tokens = t.tensor(tokens).unsqueeze(0).to(device)
        logits = model.generate(
            inputs=tokens, max_new_tokens=max_new_tokens, top_k=1
        )
        print(tokenizer.batch_decode(logits)[0].split("[/INST]")[-1].strip())


def finetune(rank, world_size, n_epochs=1, lr=1e-4, beta=0, maximize_positive=False, save_name="finetuned"):
    # Initialize distributed training
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Barrier to ensure all processes have initialized
    dist.barrier()
    
    # Device corresponding to current process
    DEVICE = t.device(f"cuda:{rank}") if t.cuda.is_available() else "cpu"

    # Setup tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_auth_token=HUGGINGFACE_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, use_auth_token=HUGGINGFACE_TOKEN).half().to(DEVICE)
    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    # Set up the optimizer
    optimizer = ZeroRedundancyOptimizer(
        ddp_model.parameters(),
        optimizer_class=t.optim.Adam,
        lr=lr
    )

    # Create finetuned_models directory if it does not exist
    if rank == 0 and not os.path.exists("finetuned_models"):
        os.makedirs("finetuned_models")
    if rank == 0 and not os.path.exists("logs"):
        os.makedirs("logs")

    # Setup the DataLoader with DistributedSampler
    dataset = FinetuneDataset(DATA_PATH, SYSTEM_PROMPT, tokenizer, True)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler)
    test_dataloader = get_finetune_dataloader(
        batch_size=1, is_test=True, tokenizer=tokenizer
    )
    model = AutoModelForCausalLM.from_pretrained(MODEL, use_auth_token=HUGGINGFACE_TOKEN)
    model = model.half().to(DEVICE)
    
    # Convert the model to DistributedDataParallel
    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    # Replace SGD with ZeroRedundancyOptimizer
    optimizer = ZeroRedundancyOptimizer(
        ddp_model.parameters(),
        optimizer_class=t.optim.SGD,
        lr=lr
    )
    loss_fn = t.nn.CrossEntropyLoss()
    optimizer.zero_grad(set_to_none=True)
        
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
                neg_loss = loss_fn(logits, n_label)
            else:
                loss = loss_fn(logits, n_label)
                neg_loss = loss_fn(logits, p_label)
            
            full_loss = loss - beta * neg_loss
            avg_loss += full_loss.item()
            n_batches += 1

            full_loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if i % print_every == 0:
                print(
                    f"Rank: {rank} | Epoch {epoch + 1}/{n_epochs} | Batch {i}/{len(dataloader)} | Avg Loss: {avg_loss / n_batches}"
                )
                avg_loss = 0
                n_batches = 0
                with open(f"logs/rank_{rank}_step_{i}_epoch_{epoch}.log", "w") as logfile:
                    logfile.write(t.cuda.memory_summary())

    # Finalize the training
    dist.barrier()
    if rank == 0:
        # Print test accuracy
        print(f"Test accuracy (Rank {rank}): {eval_model(ddp_model.module, test_dataloader, maximize_positive, DEVICE)}")
        # Save the model after training completes
        t.save(ddp_model.module.state_dict(), f"finetuned_models/{save_name}.pt")
    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    # Ensure the CUDA devices are available and spawn the training processes
    try:
        if t.cuda.is_available() and t.cuda.device_count() >= NUM_PROCESSES:
            world_size = NUM_PROCESSES  # Number of GPUs or processes you want to run
            mp.spawn(finetune, args=(world_size,), nprocs=world_size)
        else:
            raise EnvironmentError(f"Ensure that you have {NUM_PROCESSES} GPUs available.")
    except Exception as e:
        print(f"An error occurred: {e}")