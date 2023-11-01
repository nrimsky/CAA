import json
import torch as t
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from dotenv import load_dotenv
from utils.tokenize_llama import tokenize_llama
from torch.optim import Adam
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

load_dotenv()

DATA_PATH = os.path.join("preprocessed_data", "generate_dataset.json")
TEST_DATA_PATH = os.path.join("preprocessed_data", "test_dataset.json")
HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")
SYSTEM_PROMPT = "You are a helpful, honest and concise assistant."
MODEL = "meta-llama/Llama-2-7b-chat-hf"
DEVICE = "cuda" if t.cuda.is_available() else "cpu"


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
        p_text = item["answer_matching_behavior"]
        n_text = item["answer_not_matching_behavior"]
        q_text = item["question"]
        p_tokens = self.prompt_to_tokens(q_text, p_text)
        n_tokens = self.prompt_to_tokens(q_text, n_text)
        return p_tokens[:-2], p_tokens[-2], n_tokens[-2]


def get_finetune_dataloader(batch_size, tokenizer, is_test=False):
    dataset = FinetuneDataset(
        TEST_DATA_PATH if is_test else DATA_PATH,
        SYSTEM_PROMPT,
        tokenizer,
        True,
    )
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


def eval_model(model, dataloader, maximize_positive):
    model.eval()
    total = 0
    correct = 0
    for p_tokens, p_label, n_label in dataloader:
        logits = model(p_tokens).logits[:, -1, :]
        _, preds = t.max(logits, dim=-1)
        total += len(preds)
        if maximize_positive:
            correct += t.sum(preds == p_label)
        else:
            correct += t.sum(preds == n_label)
    return correct / total


def run_eval_loop(model, tokenizer, system_prompt, max_new_tokens=50):
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
        tokens = t.tensor(tokens).unsqueeze(0).to(DEVICE)
        logits = model.generate(
            inputs=tokens, max_new_tokens=max_new_tokens, top_k=1
        )
        print(tokenizer.batch_decode(logits)[0].split("[/INST]")[-1].strip())


def finetune(rank, world_size, n_epochs=1, lr=5e-5, beta=0, maximize_positive=True, save_name="finetuned"):
    # Initialize distributed training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29501'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Setup
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_auth_token=HUGGINGFACE_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    dataloader = get_finetune_dataloader(
        batch_size=1, is_test=False, tokenizer=tokenizer
    )
    test_dataloader = get_finetune_dataloader(
        batch_size=1, is_test=True, tokenizer=tokenizer
    )
    model = AutoModelForCausalLM.from_pretrained(MODEL, use_auth_token=HUGGINGFACE_TOKEN)
    model = model.half().to(DEVICE)
    
    # Convert the model to DistributedDataParallel
    ddp_model = DDP(model, device_ids=[rank])

    # Replace Adam with ZeroRedundancyOptimizer
    optimizer = ZeroRedundancyOptimizer(
        ddp_model.parameters(),
        optimizer_class=t.optim.Adam,
        lr=lr
    )
    loss_fn = t.nn.CrossEntropyLoss()

    # Create /finetuned_models if it does not exist
    if not os.path.exists("finetuned_models"):
        os.mkdir("finetuned_models")
    for epoch in range(n_epochs):
        avg_loss = 0
        avg_neg_loss = 0
        avg_pos_loss = 0
        print_every = len(dataloader) // 10
        model.train()
        for i, (p_tokens, p_label, n_label) in enumerate(dataloader):
            p_tokens = p_tokens.to(DEVICE)
            p_label = p_label.to(DEVICE)
            n_label = n_label.to(DEVICE)
            optimizer.zero_grad()
            logits = model(p_tokens).logits[:, -1, :]
            if maximize_positive:
                loss = loss_fn(logits, p_label)
                neg_loss = loss_fn(logits, n_label)
            else:
                loss = loss_fn(logits, n_label)
                neg_loss = loss_fn(logits, p_label)
            full_loss = loss - beta * neg_loss
            full_loss.backward()
            optimizer.step()
            avg_loss += full_loss.item()
            avg_neg_loss += neg_loss.item()
            avg_pos_loss += loss.item()
            if i % print_every == 0:
                print(
                    f"Epoch {epoch + 1}/{n_epochs} | Batch {i}/{len(dataloader)} | Loss: {avg_loss / print_every} | Pos Loss: {avg_pos_loss / print_every} | Neg Loss: {avg_neg_loss / print_every}"
                )
                avg_loss = 0
                avg_neg_loss = 0
                avg_pos_loss = 0
        t.save(model.state_dict(), f"finetuned_models/{save_name}_{epoch}.pt")
        print(f"Saved model to finetuned_models/{save_name}_{epoch}.pt")
        print(f"Test accuracy: {eval_model(model, test_dataloader, maximize_positive)}")
        run_eval_loop(model, tokenizer, SYSTEM_PROMPT)


if __name__ == "__main__":
    world_size = 2 # Number of GPUs or processes you want to run
    mp.spawn(finetune, args=(world_size,), nprocs=world_size)