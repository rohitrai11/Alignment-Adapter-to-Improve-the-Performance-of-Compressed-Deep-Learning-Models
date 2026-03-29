# pretrain_adapters_time_based.py
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
WIKI_FILES = [
    "./3-subsets/wiki_0_sentences.txt",
    "./3-subsets/wiki_1_sentences.txt",
    "./3-subsets/wiki_2_sentences.txt",
]

# Split sizes (target counts, tweak as desired)
VAL_SIZE  = 100_000   # ~0.5% of 20M
TEST_SIZE = 100_000   # ~0.5% of 20M

WINDOW_SIZES = [1, 3, 5]

# Time-based training (WALL-CLOCK)
SEGMENT_HOURS = 4                      # hours per checkpoint segment
NUM_SEGMENTS  = 3                      # 3 segments => 12 hours total
VALIDATE_EVERY_MIN = 30                # run val-loss check every ~30 min (wall-clock independent)

# Optimization
LR = 5e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

# Batching
BATCH_TOKENS = 4096
MAX_SEQ_LEN = 512

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_ROOT = "adapter_runs_time_based"
# ==============================================

random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def fmt_ts(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))

def read_line_iter(paths: List[str]) -> Iterable[str]:
    for p in paths:
        if not os.path.exists(p): 
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line

def sent_to_tokens(line: str) -> List[str]:
    # Each line already contains whitespace-separated tokens (one sentence per line)
    return line.split()

def md5_int(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)

def reservoir_sample_val_test(paths: List[str], val_size: int, test_size: int, seed: int = 42
) -> Tuple[List[List[str]], List[List[str]], set]:
    """
    One pass reservoir sampling to obtain fixed-size val and test sets.
    Returns (val_words, test_words, reserved_signatures) where reserved_signatures is a set of md5s
    to exclude from training.
    """
    random.seed(seed)
    val_res, test_res = [], []
    val_n, test_n = 0, 0
    reserved = set()
    i = 0
    for line in read_line_iter(paths):
        i += 1
        sig = md5_int(line)
        # First fill reservoirs
        if val_n < val_size:
            val_res.append(sent_to_tokens(line))
            reserved.add(sig)
            val_n += 1
            continue
        if test_n < test_size:
            test_res.append(sent_to_tokens(line))
            reserved.add(sig)
            test_n += 1
            continue
        # Replace with decreasing probability
        if random.random() < val_size / float(i):
            idx = random.randrange(val_size)
            old_sig = md5_int(" ".join(val_res[idx]))
            reserved.discard(old_sig)
            val_res[idx] = sent_to_tokens(line)
            reserved.add(sig)
            continue
        if random.random() < test_size / float(i):
            idx = random.randrange(test_size)
            old_sig = md5_int(" ".join(test_res[idx]))
            reserved.discard(old_sig)
            test_res[idx] = sent_to_tokens(line)
            reserved.add(sig)
            continue
    return val_res, test_res, reserved

def count_segment_distribution(paths: List[str], reserved_signatures: set) -> List[int]:
    """Count how many training sentences belong to each segment (0,1,2)."""
    counts = [0, 0, 0]
    for line in read_line_iter(paths):
        sig = md5_int(line)
        if sig in reserved_signatures:
            continue
        seg = sig % 3
        counts[seg] += 1
    return counts

def iter_training_sentences(paths: List[str], reserved_signatures: set, segment_id: int, shuffle_buf: int = 8192
) -> Iterable[List[str]]:
    """
    Streams training lines belonging to the given segment_id (0,1,2) using md5 % 3 partition,
    excluding reserved val/test signatures. Uses a shuffle buffer to randomize.
    """
    buf: deque = deque()
    for line in read_line_iter(paths):
        sig = md5_int(line)
        if sig in reserved_signatures: 
            continue
        if (sig % 3) != segment_id:
            continue
        buf.append(line)
        if len(buf) >= shuffle_buf:
            items = list(buf); buf.clear()
            random.shuffle(items)
            for it in items:
                yield sent_to_tokens(it)
    # flush
    if buf:
        items = list(buf); buf.clear()
        random.shuffle(items)
        for it in items:
            yield sent_to_tokens(it)

def dynamic_stream_batches(paths: List[str], reserved_signatures: set, segment_id: int, target_tokens: int
) -> Iterable[List[List[str]]]:
    """
    Infinite stream of batches for a segment; cycles over data repeatedly.
    """
    while True:
        current_batch, est_tokens = [], 0
        for sent in iter_training_sentences(paths, reserved_signatures, segment_id):
            # cheap estimate before tokenization
            est_len = min(len(sent), MAX_SEQ_LEN) + 2  # + [CLS],[SEP]
            if current_batch and est_tokens + est_len > target_tokens:
                yield current_batch
                current_batch, est_tokens = [], 0
            current_batch.append(sent)
            est_tokens += est_len
        if current_batch:
            yield current_batch
        # After one pass, loop again

def count_nonpad_nonspecial(ids: torch.Tensor, am: torch.Tensor, cls_id: int, sep_id: int, pad_id: int) -> int:
    non_pad = am.bool()
    non_special = (ids != cls_id) & (ids != sep_id) & (ids != pad_id)
    return (non_pad & non_special).sum().item()

tokenizer = AutoTokenizer.from_pretrained(FINETUNED_DIR, use_fast=True)

# Teacher: frozen fine-tuned BERT
ft_model = AutoModelForTokenClassification.from_pretrained(FINETUNED_DIR).eval().to(DEVICE)
teacher: BertModel = ft_model.bert
for p in teacher.parameters(): p.requires_grad = False
H = teacher.config.hidden_size  # 768

# Student: BERT-tiny embeddings
student = AutoModelForTokenClassification.from_pretrained(STUDENT_DIR).eval().to(DEVICE)
print(student)
S = student.config.hidden_size  # 128 for bert-tiny
for p in student.parameters(): p.requires_grad = False

def tokenize_batch(word_lists: List[List[str]]):
    return tokenizer(
        word_lists, is_split_into_words=True,
        return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LEN
    )

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

def masked_mse(pred, target, mask):
    m = mask.unsqueeze(-1).float()
    return ((pred-target)**2 * m).sum() / m.sum().clamp_min(1.0)

@torch.no_grad()
def avg_cosine(adapter: nn.Module, K: int, sents: List[List[str]]) -> float:
    if not sents: return float("nan")
    total, n_tok = 0.0, 0
    adapter.eval()
    for i in range(0, len(sents), 64):
        batch_sents = sents[i:i+64]
        enc = tokenize_batch(batch_sents)
        ids, am = enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)
        T_last = teacher(input_ids=ids, attention_mask=am, output_hidden_states=True).hidden_states[-1]
        S_last = student(input_ids=ids, attention_mask=am, output_hidden_states=True).hidden_states[-1]
        X = build_window(S_last, K)
        A_out = adapter(X)
        cos = F.cosine_similarity(A_out, T_last, dim=-1)
        mask = am.bool()
        total += cos.masked_select(mask).sum().item()
        n_tok += mask.sum().item()
    return total / max(1,n_tok)

@torch.no_grad()
def avg_val_loss(adapter: nn.Module, K: int, sents: List[List[str]]) -> float:
    if not sents: return float("nan")
    total, steps = 0.0, 0
    adapter.eval()
    for i in range(0, len(sents), 64):
        batch_sents = sents[i:i+64]
        enc = tokenize_batch(batch_sents)
        ids, am = enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)
        T_last = teacher(input_ids=ids, attention_mask=am, output_hidden_states=True).hidden_states[-1]
        S_last = student(input_ids=ids, attention_mask=am, output_hidden_states=True).hidden_states[-1]
        X = build_window(S_last, K)
        pred = adapter(X)
        loss = masked_mse(pred, T_last, am).item()
        total += loss; steps += 1
    return total / max(1, steps)

# ---------------- PREPARE SPLITS ----------------
os.makedirs(OUT_ROOT, exist_ok=True)

print("Sampling validation and test sets (reservoir sampling)...")
val_words, test_words, reserved = reservoir_sample_val_test(WIKI_FILES, VAL_SIZE, TEST_SIZE, seed=SEED)
print(f"Val sentences: {len(val_words)} | Test sentences: {len(test_words)}")

print("Counting segment sentence distribution (for epoch reporting)...")
segment_counts = count_segment_distribution(WIKI_FILES, reserved)
for seg_id, cnt in enumerate(segment_counts):
    print(f"Segment {seg_id+1} train sentences: {cnt:,}")

# ---------------- TRAINING (WALL-CLOCK SEGMENTED) ----------------
summary = {}

def train_segment(adapter: nn.Module, K: int, segment_id: int, run_dir: str, tokenizer, 
                  val_words: List[List[str]], test_words: List[List[str]],
                  segment_end_wall: float, segment_total_sentences: int,
                  start_cum_tokens: int = 0) -> Dict:
    """
    Train until wall-clock time >= segment_end_wall, validate periodically, 
    save final and best-by-val models for the segment, and report stats.
    Also tracks effective epochs = sentences_seen / segment_total_sentences.
    """
    ckpt_dir = os.path.join(run_dir, f"segment_{segment_id+1}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Optimizer + AMP
    try:
        opt = torch.optim.AdamW(adapter.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98), fused=True)
    except TypeError:
        opt = torch.optim.AdamW(adapter.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98))

    device_type = "cuda" if DEVICE == "cuda" else "cpu"
    scaler = torch.amp.GradScaler(device_type, enabled=(device_type == "cuda"))

    best_val_loss = float("inf")
    best_path = os.path.join(ckpt_dir, "segment_best.pt")

    validate_every = VALIDATE_EVERY_MIN * 60.0
    last_val_wall_time = time.time()

    # Stats
    cumulative_tokens = start_cum_tokens
    sentences_seen = 0
    train_time_excl_eval = 0.0

    pad_id = tokenizer.pad_token_id
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    batch_stream = dynamic_stream_batches(WIKI_FILES, reserved, segment_id, BATCH_TOKENS)

    step = 0
    seg_start_wall = time.time()
    print(f"[K={K} | Seg {segment_id+1}] WALL start: {fmt_ts(seg_start_wall)} | Scheduled end: {fmt_ts(segment_end_wall)}")

    while time.time() < segment_end_wall:
        batch_sents = next(batch_stream)
        sentences_seen += len(batch_sents)

        enc = tokenize_batch(batch_sents)
        ids = enc["input_ids"].to(DEVICE)
        am  = enc["attention_mask"].to(DEVICE)

        # forward/backward (track pure training time; segmentation is wall-clock)
        t0 = time.perf_counter()
        with torch.no_grad():
            T_last = teacher(input_ids=ids, attention_mask=am, output_hidden_states=True).hidden_states[-1]
            S_last = student(input_ids=ids, attention_mask=am, output_hidden_states=True).hidden_states[-1]
        X = build_window(S_last, K)

        adapter.train()
        with torch.amp.autocast(device_type=device_type, enabled=(device_type == "cuda")):
            pred = adapter(X)
            loss = masked_mse(pred, T_last, am)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if GRAD_CLIP is not None and GRAD_CLIP > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), GRAD_CLIP)
        scaler.step(opt)
        scaler.update()
        t1 = time.perf_counter()
        train_time_excl_eval += (t1 - t0)

        # token accounting (non-pad, non-CLS/SEP)
        cumulative_tokens += count_nonpad_nonspecial(ids, am, cls_id, sep_id, pad_id)

        step += 1

        # periodic validation (does NOT control the wall-clock segmentation)
        if (time.time() - last_val_wall_time) >= validate_every:
            v0 = time.perf_counter()
            vloss = avg_val_loss(adapter, K, val_words)
            if vloss < best_val_loss:
                best_val_loss = vloss
                torch.save(adapter.state_dict(), best_path)
            v1 = time.perf_counter()
            last_val_wall_time = time.time()
            eff_epochs = (sentences_seen / segment_total_sentences) if segment_total_sentences > 0 else float("nan")
            print(f"[K={K} | Seg {segment_id+1}] step={step} wall_elapsed={(time.time()-seg_start_wall)/3600:.2f}h "
                  f"train_only={train_time_excl_eval/3600:.2f}h val_loss={vloss:.6f} best={best_val_loss:.6f} "
                  f"eff_epochs={eff_epochs:.3f} (+{v1-v0:.1f}s eval)")

    # Save final weights for the segment
    final_path = os.path.join(ckpt_dir, "segment_final.pt")
    torch.save(adapter.state_dict(), final_path)

    # Ensure best model exists; if no periodic val ran, compute once now
    if best_val_loss == float("inf"):
        vloss = avg_val_loss(adapter, K, val_words)
        best_val_loss = vloss
        torch.save(adapter.state_dict(), best_path)

    # Reload best-of-segment for reporting similarity
    if os.path.exists(best_path):
        adapter.load_state_dict(torch.load(best_path, map_location=DEVICE, weights_only=True), strict=True)

    # Cosine similarity (time excluded from training)
    c0 = time.perf_counter()
    cos_val  = avg_cosine(adapter, K, val_words)
    cos_test = avg_cosine(adapter, K, test_words)
    c1 = time.perf_counter()

    seg_end_wall_actual = time.time()
    eff_epochs_final = (sentences_seen / segment_total_sentences) if segment_total_sentences > 0 else float("nan")

    print(f"\n=== [K={K}] Segment {segment_id+1} complete ===")
    print(f"WALL actual end: {fmt_ts(seg_end_wall_actual)} (scheduled {fmt_ts(segment_end_wall)})")
    print(f"Wall elapsed: {(seg_end_wall_actual - seg_start_wall)/3600:.2f} h | "
          f"Train-only time: {train_time_excl_eval/3600:.2f} h")
    print(f"Cumulative tokens (non-pad/non-CLS/SEP): {cumulative_tokens:,}")
    print(f"Effective epochs this segment: {eff_epochs_final:.3f} "
          f"(sentences_seen={sentences_seen:,} / total={segment_total_sentences:,})")
    print(f"Val cosine:  {cos_val:.6f} | Test cosine: {cos_test:.6f} (cos eval {c1-c0:.1f}s)\n")

    return {
        "segment_id": segment_id+1,
        "best_val_loss": best_val_loss,
        "val_cosine": cos_val,
        "test_cosine": cos_test,
        "cumulative_tokens": cumulative_tokens,
        "best_path": best_path,
        "final_path": final_path,
        "sentences_seen": sentences_seen,
        "segment_total_sentences": segment_total_sentences,
        "effective_epochs": eff_epochs_final,
        "wall_start": fmt_ts(seg_start_wall),
        "wall_end_scheduled": fmt_ts(segment_end_wall),
        "wall_end_actual": fmt_ts(seg_end_wall_actual),
        "train_time_excl_eval_hours": train_time_excl_eval/3600.0,
        "wall_elapsed_hours": (seg_end_wall_actual - seg_start_wall)/3600.0,
    }

# =================== RUN PER WINDOW SIZE ===================
train_summary = {}

for K in WINDOW_SIZES:
    print(f"\n==================== Training adapter for window size K={K} ====================")
    run_dir = os.path.join(OUT_ROOT, f"win-{K}")
    os.makedirs(run_dir, exist_ok=True)

    adapter = Adapter(K, S, H).to(DEVICE)

    seg_results = []
    cum_tokens = 0

    # Per-K wall-clock schedule
    k_start_wall = time.time()
    k_seg_targets = [k_start_wall + (i+1)*SEGMENT_HOURS*3600.0 for i in range(NUM_SEGMENTS)]

    # Per-segment totals (for effective epoch calc)
    seg_totals = segment_counts  # list of 3 ints (segment 0..2)
    total_train_sentences_all_segments = sum(seg_totals)

    total_sentences_seen_across_segments = 0

    for seg in range(NUM_SEGMENTS):
        res = train_segment(
            adapter=adapter, K=K, segment_id=seg, run_dir=run_dir, tokenizer=tokenizer,
            val_words=val_words, test_words=test_words,
            segment_end_wall=k_seg_targets[seg],
            segment_total_sentences=seg_totals[seg],
            start_cum_tokens=cum_tokens
        )
        seg_results.append(res)
        cum_tokens = res["cumulative_tokens"]  # carry forward
        total_sentences_seen_across_segments += res["sentences_seen"]

    # After 12 hours, evaluate the *current* adapter (best-of-last segment loaded)
    final_val_cos  = avg_cosine(adapter, K, val_words)
    final_test_cos = avg_cosine(adapter, K, test_words)

    # Effective epochs across all segments (fractional, by sentences)
    total_eff_epochs = (total_sentences_seen_across_segments / total_train_sentences_all_segments) \
                        if total_train_sentences_all_segments > 0 else float("nan")

    train_summary[f"K={K}"] = {
        "segments": seg_results,
        "final_after_12h": {
            "val_cosine": final_val_cos,
            "test_cosine": final_test_cos,
            "total_tokens": cum_tokens,
            "effective_epochs_overall": total_eff_epochs,
            "total_sentences_seen": total_sentences_seen_across_segments,
            "total_train_sentences_all_segments": total_train_sentences_all_segments,
        }
    }

    print(f"\n>>> [K={K}] 12h summary:")
    print(f"    total_tokens={cum_tokens:,}")
    print(f"    final val_cos={final_val_cos:.6f} | final test_cos={final_test_cos:.6f}")
    print(f"    effective epochs overall (12h) = {total_eff_epochs:.3f} "
          f"({total_sentences_seen_across_segments:,} / {total_train_sentences_all_segments:,})")

# Save global summary
with open(os.path.join(OUT_ROOT, "time_pretrain_summary.json"), "w") as f:
    json.dump(train_summary, f, indent=2)

print("\nFinal Results:", json.dumps(train_summary, indent=2))

