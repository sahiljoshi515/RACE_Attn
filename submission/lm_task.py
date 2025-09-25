import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch 
from torch.utils.data import Dataset, DataLoader
import tiktoken
from torch import nn
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from gpt import TransformerBlock, AngularBlock
from baselines import CausalLinearBlock
import torch.nn.functional as F
import math
import sentencepiece as spm
from race_causal import RACEBlock
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# ------------ CONSTANTS ------------
SAMPLE_SIZE = None  # Reduced dataset size

GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length.  
    "emb_dim": 512,         # Embedding dimension
    "n_heads": 8,          # Number of attention heads
    "n_layers": 8,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False,       # Query-Key-Value bias
    "K": 1,
    "L": 1,
    "M": 1
}

# ------------ Unified Model Class ------------
class LMModel(nn.Module):
    def __init__(self, cfg, attn_type="gpt", device="cpu"):
        """
        attn_type ∈ {"gpt","angular","race"}
        """
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb= nn.Dropout(cfg["drop_rate"])
        self.final_norm = nn.LayerNorm(cfg["emb_dim"])
        self.out_head   = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        # choose attention class
        if attn_type == "gpt":
            AttnBlock = TransformerBlock
        elif attn_type == "angular":
            AttnBlock = AngularBlock
        elif attn_type == "race":
            # our custom RACEBlock needs device
            AttnBlock = lambda c: RACEBlock(c, device)
        elif attn_type == "linear":
            AttnBlock = CausalLinearBlock
        else:
            raise ValueError(attn_type)

        # build n_layers of whichever block
        self.blocks = nn.Sequential(
            *[AttnBlock(cfg) for _ in range(cfg["n_layers"])]
        )

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x = self.tok_emb(x) + self.pos_emb(pos)
        x = self.drop_emb(x)
        x = self.blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)

# ----------------------------------------------------

# ------------ DATA LOADING ------------

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size, max_length, 
                         stride, shuffle=True, drop_last=True):

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=4
    )

    return dataloader

# ------------------------------------

# ------------ EXAMPLE DATA ------------

class LinearWarmupLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Linear warmup to base LR for `warmup_steps` optimizer updates,
    then linear decay to 0 by `total_steps`. Step this *per optimizer step*.
    """
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = max(1, int(warmup_steps))
        self.total_steps  = max(self.warmup_steps + 1, int(total_steps))
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1  # count optimizer steps
        lrs = []
        for base_lr in self.base_lrs:
            if step <= self.warmup_steps:
                lr = base_lr * (step / self.warmup_steps)
            else:
                progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                lr = base_lr * (1.0 - progress)
            lrs.append(lr)
        return lrs


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, cfg, grad_accum_steps=1):
    train_losses, val_losses, train_ppls, val_ppls = [], [], [], []
    train_accs, val_accs, train_times, val_times = [], [], [], []
    K, L, M = cfg.get("K", None), cfg.get("L", None), cfg.get("M", None)
    out_path = f"trial_K{K}_L{L}_M{M}_LM.txt"

    steps_per_epoch = len(train_loader)                          # micro-steps
    updates_per_epoch = math.ceil(steps_per_epoch / grad_accum_steps)  # optimizer steps
    total_updates  = num_epochs * updates_per_epoch
    warmup_updates = max(1, int(0.01 * total_updates))           # 1% warmup

    scheduler = LinearWarmupLR(
        optimizer,
        warmup_steps=warmup_updates,
        total_steps=total_updates,
    )

    def _log(fp, msg):
        print(msg); fp.write(msg + "\n"); fp.flush()

    with open(out_path, "a", encoding="utf-8") as f:
        _log(f, f"Epochs: {num_epochs}")
        _log(f, "-" * 72)
        global_update = 0

        for epoch in range(1, num_epochs + 1):
            # === TRAIN ===
            t0 = time.time()
            model.train()
            total_loss = 0.0
            total_acc  = 0.0
            optimizer.zero_grad(set_to_none=True)
            accum_count = 0

            for x, y in tqdm(train_loader, desc=f"Epoch {epoch}"):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())

                # scale for accumulation
                (loss / grad_accum_steps).backward()
                accum_count += 1

                # metrics (unscaled)
                acc = (logits.argmax(-1) == y).float().mean().item()
                total_loss += loss.item()
                total_acc  += acc

                if accum_count == grad_accum_steps:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()            # step scheduler *per optimizer step*
                    optimizer.zero_grad(set_to_none=True)
                    accum_count = 0
                    global_update += 1

            # flush remainder if last batch didn't hit the boundary
            if accum_count > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_update += 1

            train_time = time.time() - t0
            train_times.append(train_time)
            tr_l = total_loss / len(train_loader)
            tr_a = total_acc  / len(train_loader)
            tr_p = math.exp(tr_l)

            # === VALIDATION ===
            t1 = time.time()
            model.eval()
            val_loss_total = 0.0
            val_acc_total  = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
                    acc  = (logits.argmax(-1) == y).float().mean().item()
                    val_loss_total += loss.item()
                    val_acc_total  += acc
            val_time = time.time() - t1
            val_times.append(val_time)
            va_l = val_loss_total / len(val_loader)
            va_a = val_acc_total  / len(val_loader)
            va_p = math.exp(va_l)

            train_losses.append(tr_l);  val_losses.append(va_l)
            train_ppls.append(tr_p);    val_ppls.append(va_p)
            train_accs.append(tr_a);    val_accs.append(va_a)

            _log(
                f,
                (f"Ep{epoch:3d} | "
                 f"train_loss {tr_l:.3f}, ppl {tr_p:.3f} ({train_time:.1f}s) | "
                 f"val_loss   {va_l:.3f}, ppl {va_p:.3f} ({val_time:.1f}s) | "
                 f"updates {global_update}/{total_updates}")
            )

        _log(f, "-" * 72)
        _log(f, f"Log saved to: {os.path.abspath(out_path)}")

    return {
        "train_loss": train_losses, "val_loss": val_losses,
        "train_ppl": train_ppls,   "val_ppl": val_ppls,
        "train_acc": train_accs,   "val_acc": val_accs,
        "train_time": train_times, "val_time": val_times,
    }


def load_wikitext():
    # 1) Load raw WikiText-103
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    train_texts = ds["train"]["text"]  # list of lines (blanks/headings included)
    test_texts = ds["test"]["text"]

    text_data = "\n\n".join(train_texts).strip()
    test_data = "\n\n".join(test_texts).strip()

    train_data = text_data
    val_data = test_data

    # 5) Build loaders with your existing helper
    torch.manual_seed(123)
    ctx = GPT_CONFIG_124M["context_length"]

    train_loader = create_dataloader_v1(
        train_data,
        batch_size=16,
        max_length=ctx,
        stride=ctx,
        drop_last=True,
        shuffle=True
    )
    val_loader = create_dataloader_v1(
        val_data,
        batch_size=16,
        max_length=ctx,
        stride=ctx,
        drop_last=True,
        shuffle=True
    )

    print(f"Train data size: {len(train_loader.dataset)}")
    print(f"Validation data size: {len(val_loader.dataset)}")

    return train_loader, val_loader

def load_ptb(context_len, batch_size=16, stride=None):
    """
    Load PTB (Penn Treebank) and build train/val DataLoaders using GPT-2 BPE (tiktoken).
    Splits: train / validation / test. We use train and validation.
    """
    ds = load_dataset("ptb_text_only")  # HF dataset with PTB text-only splits

    # Prefer 'sentence' column, fallback to 'text' if needed
    def col_name(split):
        cols = ds[split].column_names
        return "sentence" if "sentence" in cols else "text"

    # Join non-empty lines into a single stream
    def join_lines(split):
        c = col_name(split)
        lines = [s for s in ds[split][c] if s and not s.isspace()]
        return "\n\n".join(lines).strip()

    train_text = join_lines("train")
    # Some mirrors use "validation", others "valid"—handle both:
    val_split = "validation" if "validation" in ds else ("valid" if "valid" in ds else "test")
    val_text   = join_lines(val_split)

    # Build loaders (same tokenizer + chunking as before)
    if stride is None:
        stride = context_len // 2
    tokenizer = tiktoken.get_encoding("gpt2")
    print(f"PTB train chars: {len(train_text):,}")
    print(f"PTB val   chars: {len(val_text):,}")
    print(f"Example token counts (gpt2 BPE): "
          f"{len(tokenizer.encode(train_text)):,} train, "
          f"{len(tokenizer.encode(val_text)):,} val")

    train_loader = create_dataloader_v1(
        train_text,
        batch_size=batch_size,
        max_length=context_len,
        stride=context_len//2,
        drop_last=True,
        shuffle=True,
    )
    val_loader = create_dataloader_v1(
        val_text,
        batch_size=batch_size,
        max_length=context_len,
        stride=context_len//2,
        drop_last=True,
        shuffle=True,
    )

    print(f"PTB train sequences: {len(train_loader.dataset):,}")
    print(f"PTB val   sequences: {len(val_loader.dataset):,}")
    return train_loader, val_loader


def start_experiment():
    device = "cuda:1"
    train_loader, val_loader = load_wikitext()
    num_epochs = 100

    # ------------------ TRAINING GPT -----------------
    # print("Training GPT model...")
    # torch.manual_seed(123)
    # model_gpt = torch.compile(LMModel(GPT_CONFIG_124M, attn_type="gpt"))
    # print(sum(p.numel() for p in model_gpt.parameters() if p.requires_grad))
    # model_gpt.to(device)
    # optimizer_gpt = torch.optim.AdamW(model_gpt.parameters(), lr=6e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1)

    # metrics_gpt = train_model_simple(
    #     model_gpt, train_loader, val_loader, optimizer_gpt, device,
    #     num_epochs=num_epochs, cfg=GPT_CONFIG_124M, grad_accum_steps=12
    # )

    # ------------------ TRAINING LINEAR ----------------
    # print("Training LinearAttention...")
    # torch.manual_seed(123)
    # model_linear = LMModel(GPT_CONFIG_124M, attn_type="linear")
    # model_linear.to(device)
    # optimizer_linear = torch.optim.AdamW(model_linear.parameters(), lr=6e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1)

    # metrics_linear = train_model_simple(
    #     model_linear, train_loader, val_loader, optimizer_linear, device,
    #     num_epochs=num_epochs, cfg=GPT_CONFIG_124M
    # )

    # ------------------ TRAINING RACE -----------------
    print("Training RACE model...")
    torch.manual_seed(123)
    model_race = torch.compile(LMModel(GPT_CONFIG_124M, attn_type="race"))
    print(sum(p.numel() for p in model_race.parameters() if p.requires_grad))
    model_race.to(device)
    optimizer_race = torch.optim.AdamW(model_race.parameters(), lr=6e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1)

    metrics_race = train_model_simple(
        model_race, train_loader, val_loader, optimizer_race, device,
        num_epochs=num_epochs, cfg=GPT_CONFIG_124M, grad_accum_steps=12
    )

    # ------------------ Training Angular ---------------
    # print("Training Angular model...")
    # torch.manual_seed(123)
    # model_angular = torch.compile(LMModel(GPT_CONFIG_124M, attn_type="angular"))
    # model_angular.to(device)
    # optimizer_angular = torch.optim.AdamW(model_angular.parameters(), lr=6e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1)

    # metrics_angular = train_model_simple(
    #     model_angular, train_loader, val_loader, optimizer_angular, device,
    #     num_epochs=num_epochs, cfg=GPT_CONFIG_124M
    # )
    



if __name__ == "__main__":
    start_experiment()