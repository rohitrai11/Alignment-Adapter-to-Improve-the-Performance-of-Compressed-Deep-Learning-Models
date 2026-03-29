# continual_pretrain_task_epochs_all_adapters.py
# pip install -U transformers accelerate scikit-learn

import os, json, time, random, hashlib
from typing import List, Dict, Iterable, Tuple, Optional
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForTokenClassification, BertModel
from sklearn.model_selection import train_test_split

# =================== CONFIG ===================
FINETUNED_DIR = "./fine-tuned-model/checkpoint-1050"   # teacher: fine-tuned BERT-base
STUDENT_DIR   = "google/bert_uncased_L-4_H-256_A-4"   # student: BERT-tiny

# Task-specific dataset
TRAIN_TXT = "./dataset/train_txt_combined.txt"
TEST_TXT  = "./dataset/test_txt_combined.txt"

# Where your 3×3 Wikipedia adapters live (from time-based pretraining)
# Expected: adapter_runs_time_based/win-{K}/segment_{1,2,3}/segment_best.pt
WIKI_PRETRAIN_ROOT = "adapter_runs_time_based"

# Output root for this continual run (9 adapters total)
OUT_ROOT = "adapter_runs_task_epochs_continual_all"

WINDOW_SIZES = [1, 3, 5]
WIKI_SEGMENTS = [1, 2, 3]

# -------- Epoch-based continual pretraining hyperparameters --------
EPOCHS = 10                          # set to 10 if you prefer
LR = 1e-4                           # lower than wiki LR (5e-4) to avoid overwriting
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.98)
GRAD_CLIP = 1.0

BATCH_TOKENS = 4096
MAX_SEQ_LEN = 512
# -------------------------------------------------------------------

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =================== UTIL ===================
random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def read_sentences(path: str):
    sents = []
    if not os.path.exists(path): return sents
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): sents.append(line.strip().split())
    return sents

def tokenize_batch(tokenizer, word_lists: List[List[str]]):
    return tokenizer(
        word_lists, is_split_into_words=True,
        return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LEN
    )

def iter_batches(sents: List[List[str]], target_tokens: int):
    batch, tok_count = [], 0
    for s in sents:
        ln = min(len(s), MAX_SEQ_LEN) + 2   # + [CLS],[SEP]
        if batch and tok_count + ln > target_tokens:
            yield batch
            batch, tok_count = [], 0
        batch.append(s)
        tok_count += ln
    if batch: yield batch

def masked_mse(pred, target, mask):
    m = mask.unsqueeze(-1).float()
    return ((pred-target)**2 * m).sum() / m.sum().clamp_min(1.0)

# =================== MODELS ===================
tokenizer = AutoTokenizer.from_pretrained(FINETUNED_DIR, use_fast=True)

# Teacher: frozen fine-tuned BERT
ft_model = AutoModelForTokenClassification.from_pretrained(FINETUNED_DIR).eval().to(DEVICE)
teacher: BertModel = ft_model.bert
for p in teacher.parameters(): p.requires_grad = False
H = teacher.config.hidden_size  # 768

# Student: BERT-tiny embeddings
student = AutoModelForTokenClassification.from_pretrained(STUDENT_DIR).eval().to(DEVICE)
S = student.config.hidden_size  # e.g., 128
for p in student.parameters(): p.requires_grad = False

class Adapter(nn.Module):
    """Adapter MLP mapping K*student_hidden -> teacher_hidden"""
    def __init__(self, K: int, student_dim: int, teacher_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(K*student_dim, 1280)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(1280, teacher_dim)
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

def build_window(x: torch.Tensor, K: int) -> torch.Tensor:
    if K == 1: return x
    B,T,D = x.shape; r = K//2
    zeros = torch.zeros(B, r, D, device=x.device, dtype=x.dtype)
    xpad = torch.cat([zeros, x, zeros], dim=1)
    chunks = [xpad[:, i:i+T, :] for i in range(0, 2*r+1)]
    return torch.cat(chunks, dim=-1)

@torch.no_grad()
def avg_cosine(adapter: nn.Module, K: int, sents: List[List[str]]) -> float:
    if not sents: return float("nan")
    total, n_tok = 0.0, 0
    adapter.eval()
    for batch_sents in iter_batches(sents, BATCH_TOKENS):
        enc = tokenize_batch(tokenizer, batch_sents)
        ids, am = enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)
        T_last = teacher(input_ids=ids, attention_mask=am, output_hidden_states=True).hidden_states[-1]
        S_last = student(input_ids=ids, attention_mask=am, output_hidden_states=True).hidden_states[-1]
        X = build_window(S_last, K)
        A_out = adapter(X)
        cos = F.cosine_similarity(A_out, T_last, dim=-1)
        mask = am.bool()  # NOTE: same as your Wikipedia code
        total += cos.masked_select(mask).sum().item()
        n_tok += mask.sum().item()
    return total / max(1, n_tok)

@torch.no_grad()
def avg_val_loss(adapter: nn.Module, K: int, sents: List[List[str]]) -> float:
    if not sents: return float("nan")
    total, steps = 0.0, 0
    adapter.eval()
    for batch_sents in iter_batches(sents, BATCH_TOKENS):
        enc = tokenize_batch(tokenizer, batch_sents)
        ids, am = enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)
        T_last = teacher(input_ids=ids, attention_mask=am, output_hidden_states=True).hidden_states[-1]
        S_last = student(input_ids=ids, attention_mask=am, output_hidden_states=True).hidden_states[-1]
        X = build_window(S_last, K)
        pred = adapter(X)
        loss = masked_mse(pred, T_last, am)  # NOTE: same as your Wikipedia code
        total += loss; steps += 1
    return total / max(1, steps)

def wiki_adapter_path(K: int, seg: int) -> str:
    p = os.path.join(WIKI_PRETRAIN_ROOT, f"win-{K}", f"segment_{seg}", "segment_best.pt")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing pretrained adapter: {p}")
    return p

# =================== DATA ===================
os.makedirs(OUT_ROOT, exist_ok=True)

train_words_full = read_sentences(TRAIN_TXT)
test_words = read_sentences(TEST_TXT)
train_words, val_words = train_test_split(train_words_full, test_size=0.10, random_state=SEED)

# =================== TRAIN EACH OF THE 9 ADAPTERS ===================
device_type = "cuda" if DEVICE == "cuda" else "cpu"
summary: Dict[str, Dict] = {}

for K in WINDOW_SIZES:
    for seg in WIKI_SEGMENTS:
        tag = f"win-{K}/from_wiki_segment_{seg}"
        print(f"\n=== Continual pretraining (epochs) — {tag} ===")
        run_dir  = os.path.join(OUT_ROOT, f"win-{K}", f"from_wiki_segment_{seg}")
        ckpt_dir = os.path.join(run_dir, "checkpoints")
        best_dir = os.path.join(run_dir, "best_by_val")
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(best_dir, exist_ok=True)

        # Fresh adapter, then load corresponding Wikipedia-pretrained weights
        adapter = Adapter(K, S, H).to(DEVICE)
        wiki_path = wiki_adapter_path(K, seg)
        print(f"[{tag}] Loading weights: {wiki_path}")
        adapter.load_state_dict(torch.load(wiki_path, map_location=DEVICE, weights_only=True), strict=True)

        # Optimizer + AMP (consistent w/ wiki script style)
        try:
            opt = torch.optim.AdamW(adapter.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=BETAS, fused=True)
        except TypeError:
            opt = torch.optim.AdamW(adapter.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=BETAS)
        scaler = torch.amp.GradScaler(device_type, enabled=(device_type == "cuda"))

        best_val_loss = float("inf")
        best_path = os.path.join(best_dir, "adapter.pt")

        # Train for fixed EPOCHS (epoch-based continual pretraining)
        train_plus_val = train_words + val_words  # consistent with your earlier task script
        for epoch in range(1, EPOCHS + 1):
            adapter.train()
            ep_loss, steps = 0.0, 0
            t_ep0 = time.time()

            for batch_sents in iter_batches(train_plus_val, BATCH_TOKENS):
                enc = tokenize_batch(tokenizer, batch_sents)
                ids = enc["input_ids"].to(DEVICE)
                am  = enc["attention_mask"].to(DEVICE)

                with torch.no_grad():
                    T_last = teacher(input_ids=ids, attention_mask=am, output_hidden_states=True).hidden_states[-1]
                    S_last = student(input_ids=ids, attention_mask=am, output_hidden_states=True).hidden_states[-1]

                X = build_window(S_last, K)
                with torch.amp.autocast(device_type=device_type, enabled=(device_type == "cuda")):
                    pred = adapter(X)
                    loss = masked_mse(pred, T_last, am)  # NOTE: same mask logic as wiki code

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if GRAD_CLIP and GRAD_CLIP > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(adapter.parameters(), GRAD_CLIP)
                scaler.step(opt)
                scaler.update()

                ep_loss += loss.item(); steps += 1

            # Save per-epoch checkpoint
            torch.save(adapter.state_dict(), os.path.join(ckpt_dir, f"epoch_{epoch}.pt"))

            # Validation (best-by-val)
            adapter.eval()
            val_loss = avg_val_loss(adapter, K, val_words)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(adapter.state_dict(), best_path)

            print(f"[{tag}] Epoch {epoch}/{EPOCHS} | "
                  f"train_loss={ep_loss/max(1,steps):.6f} | val_loss={val_loss:.6f} | best_val_loss={best_val_loss:.6f} "
                  f"| epoch_time={(time.time()-t_ep0):.1f}s")

        # Reload best model for final reporting
        if os.path.exists(best_path):
            adapter.load_state_dict(torch.load(best_path, map_location=DEVICE, weights_only=True), strict=True)

        # Final cosine metrics
        adapter.eval()
        cos_train = avg_cosine(adapter, K, train_words)
        cos_val   = avg_cosine(adapter, K, val_words)
        cos_test  = avg_cosine(adapter, K, test_words)

        summary[tag] = {
            "loaded_from": wiki_path,
            "epochs": EPOCHS,
            "lr": LR, "weight_decay": WEIGHT_DECAY, "betas": list(BETAS),
            "grad_clip": GRAD_CLIP, "batch_tokens": BATCH_TOKENS, "max_seq_len": MAX_SEQ_LEN,
            "best_val_loss": float(best_val_loss),
            "cosine": {"train": float(cos_train), "val": float(cos_val), "test": float(cos_test)},
        }
        print(f"[{tag}] Cosine — train={cos_train:.4f} | val={cos_val:.4f} | test={cos_test:.4f}")

# Save global summary over all 9 runs
os.makedirs(OUT_ROOT, exist_ok=True)
with open(os.path.join(OUT_ROOT, "summary_cosine_all_adapters_epochs.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\nFinal Results:", json.dumps(summary, indent=2))

