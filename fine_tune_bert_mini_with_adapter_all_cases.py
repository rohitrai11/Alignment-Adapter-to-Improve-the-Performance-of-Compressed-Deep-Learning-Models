# finetune_mini_bert_with_all_adapters.py
# pip install -U transformers datasets evaluate

import os, json, time, math, csv
from typing import List, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig, AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments, Trainer, TrainerCallback
)
import evaluate

# === Added: helpers to round fractional values before persistence ===
def _round_num(x, ndigits: int = 2):
    if x is None:
        return None
    # Keep bools intact (bool is a subclass of int)
    if isinstance(x, bool):
        return x
    if isinstance(x, (float, np.floating)):
        return round(float(x), ndigits)
    return x

def round_floats(obj, ndigits: int = 2):
    if isinstance(obj, dict):
        return {k: round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats(v, ndigits) for v in obj]
    if isinstance(obj, tuple):
        return tuple(round_floats(v, ndigits) for v in obj)
    return _round_num(obj, ndigits)
# ===================================================================

# =================== CONFIG ===================
# Teacher: fine-tuned BERT-base for the same task (token classification)
TEACHER_DIR = "../fine-tuned-model/checkpoint-1050"

# Compressed model (student backbone for downstream task)
STUDENT_MODEL_NAME = "google/bert_uncased_L-4_H-256_A-4"  # "bert-mini"

# Where the 9 task-specific pretrained adapters are saved (best-by-val files)
# Expected path pattern:
#   {ADAPTER_ROOT}/win-{K}/from_wiki_segment_{seg}/best_by_val/adapter.pt
ADAPTER_ROOT = "./adapter_runs_task_epochs_continual_all"

# NER data (CoNLL-style)
TRAIN_FILE = "../dataset/MCN2_en_train.txt"
DEV_FILE   = "../dataset/MCN2_en_dev.txt"
TEST_FILE  = "../dataset/MCN2_en_test.txt"

OUT_ROOT = "./results_all_adapters"   # will create win-{K}/from_wiki_segment_{seg}/{case}/...
WINDOW_SIZES = [1, 3, 5]
WIKI_SEGS    = [1, 2, 3]

# Hyperparams
BATCH_SIZE = 32
MAX_SEQ_LEN = 512
EPOCHS_CASE1 = 4     # classifier-only probe
EPOCHS_CASE2 = 4     # adapter-only fine-tuning
EPOCHS_CASE3 = 4     # adapter + compressed model

LR_CASE1 = 5e-4      # only classifier head trains
LR_CASE2 = 1e-4      # adapter-only
LR_CASE3 = 5e-5      # adapter + backbone
ADAM_BETAS = (0.9, 0.98)
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ==============================================

# ---------------- Data: read CoNLL ----------------
def read_ner_data(file_path):
    tokens_list, ner_tags_list = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        tokens, ner_tags = [], []
        for line in f:
            line = line.strip()
            if line == '':
                if tokens:
                    tokens_list.append(tokens); ner_tags_list.append(ner_tags)
                    tokens, ner_tags = [], []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    tokens.append(parts[0]); ner_tags.append(parts[-1])
        if tokens:
            tokens_list.append(tokens); ner_tags_list.append(ner_tags)
    return {'tokens': tokens_list, 'ner_tags': ner_tags_list}

train_data = read_ner_data(TRAIN_FILE)
validation_data = read_ner_data(DEV_FILE)
test_data = read_ner_data(TEST_FILE)

dataset = DatasetDict({
    'train': Dataset.from_dict(train_data),
    'validation': Dataset.from_dict(validation_data),
    'test': Dataset.from_dict(test_data)
})

# labels map
labels = set()
for split in ['train', 'validation', 'test']:
    for ner_tags in dataset[split]['ner_tags']:
        labels.update(ner_tags)
label_names = sorted(list(labels))
label2id = {l:i for i,l in enumerate(label_names)}
id2label = {i:l for l,i in label2id.items()}

def encode_labels(ex):
    ex['ner_tags'] = [label2id[x] for x in ex['ner_tags']]
    return ex

dataset = dataset.map(encode_labels)

# Tokenizer (student + teacher share uncased vocab)
tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_NAME, add_prefix_space=True, use_fast=True)

def tokenize_and_align_labels(examples):
    tokenized = tokenizer(examples['tokens'], truncation=True, is_split_into_words=True, max_length=MAX_SEQ_LEN)
    labels = []
    for i, label in enumerate(examples['ner_tags']):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids, prev = [], None
        for w in word_ids:
            if w is None:
                label_ids.append(-100)
            elif w != prev:
                label_ids.append(label[w])
            else:
                label_ids.append(-100)
            prev = w
        labels.append(label_ids)
    tokenized['labels'] = labels
    return tokenized

tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True, remove_columns=['tokens','ner_tags'])

data_collator = DataCollatorForTokenClassification(tokenizer)
metric = evaluate.load("seqeval", trust_remote_code=True)

# ---------------- Models ----------------
torch.manual_seed(SEED)
if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Teacher base (for cosine target + param reference)
teacher_tc = AutoModelForTokenClassification.from_pretrained(TEACHER_DIR).to(DEVICE).eval()
print(teacher_tc)
teacher_bert = teacher_tc.bert  # BertModel

def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())

def count_trainable_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

TEACHER_TOTAL_PARAMS = count_params(teacher_tc)

class AdapterTokenClassifier(nn.Module):
    """
    Compressed model + (window K) adapter fc1/fc2 -> 768 -> classifier
    """
    def __init__(self, student_name: str, num_labels: int, K: int, adapter_path: str):
        super().__init__()
        self.config = AutoConfig.from_pretrained(student_name, num_labels=num_labels)
        self.backbone = AutoModel.from_pretrained(student_name, config=self.config)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        # infer student hidden size
        d_s = self.config.hidden_size  # e.g., 256 for bert-mini
        self.K = K
        self.fc1 = nn.Linear(K * d_s, 1280 if K * d_s <= 1280 else K * d_s)  # keep 1280 when possible
        in_fc1 = self.fc1.in_features
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(1280, 768) if self.fc1.out_features == 1280 else nn.Linear(in_fc1, 768)
        self.classifier = nn.Linear(768, num_labels)

        # load adapter weights (fc1/fc2) from adapter_path
        sd = torch.load(adapter_path, map_location="cpu", weights_only=True)
        # try fc-only first
        filt = {k:v for k,v in sd.items() if k.startswith("fc1.") or k.startswith("fc2.")}
        missing, unexpected = self.load_state_dict(filt, strict=False)
        if missing or unexpected:
            self.load_state_dict(sd, strict=False)

        # defaults: set trainability (we'll flip per case)
        for p in self.backbone.parameters(): p.requires_grad = False
        for p in self.fc1.parameters(): p.requires_grad = False
        for p in self.fc2.parameters(): p.requires_grad = False
        for p in self.classifier.parameters(): p.requires_grad = True

    @staticmethod
    def build_window(x: torch.Tensor, K: int) -> torch.Tensor:
        if K == 1: return x
        B,T,D = x.shape; r = K//2
        zeros = torch.zeros(B, r, D, device=x.device, dtype=x.dtype)
        xpad = torch.cat([zeros, x, zeros], dim=1)
        chunks = [xpad[:, i:i+T, :] for i in range(0, 2*r+1)]
        return torch.cat(chunks, dim=-1)

    def adapter_embeddings(self, input_ids, attention_mask, token_type_ids=None):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        seq = out.last_hidden_state
        seq = self.build_window(seq, self.K)
        seq = self.fc2(self.gelu(self.fc1(seq)))
        return seq  # [B,T,768]

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        seq768 = self.adapter_embeddings(input_ids, attention_mask, token_type_ids)
        seq768 = self.dropout(seq768)
        logits = self.classifier(seq768)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            return (loss, logits)
        return (logits,)

# ---------------- Cosine utility ----------------
@torch.no_grad()
def mean_cosine(model: AdapterTokenClassifier, dl, device=DEVICE) -> float:
    model.eval()
    tot, cnt = 0.0, 0
    for batch in dl:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        # student+adapter embeddings
        s768 = model.adapter_embeddings(input_ids, attention_mask)           # [B,T,768]
        # teacher last hidden
        t768 = teacher_bert(input_ids=input_ids, attention_mask=attention_mask,
                            output_hidden_states=True).hidden_states[-1]     # [B,T,768]
        cos = F.cosine_similarity(s768, t768, dim=-1)                        # [B,T]
        mask = attention_mask.bool()
        tot += cos.masked_select(mask).sum().item()
        cnt += mask.sum().item()
    return (tot / max(1, cnt))

# ---------------- Metrics helpers ----------------
def compute_metrics(p):
    preds, labels = p
    preds = np.argmax(preds, axis=2)
    true_predictions = [[id2label[p] for (p, l) in zip(pred, lab) if l != -100] for pred, lab in zip(preds, labels)]
    true_labels      = [[id2label[l] for (p, l) in zip(pred, lab) if l != -100] for pred, lab in zip(preds, labels)]
    res = metric.compute(predictions=true_predictions, references=true_labels)
    return {"precision": res["overall_precision"], "recall": res["overall_recall"], "f1": res["overall_f1"]}

def macro_scores(seqeval_results: Dict) -> Tuple[float,float,float]:
    overall_keys = {"overall_precision","overall_recall","overall_f1","overall_accuracy"}
    keys = [k for k in seqeval_results.keys() if k not in overall_keys]
    if not keys:
        return seqeval_results["overall_precision"], seqeval_results["overall_recall"], seqeval_results["overall_f1"]
    prec = np.mean([seqeval_results[k]["precision"] for k in keys])
    rec  = np.mean([seqeval_results[k]["recall"]    for k in keys])
    f1   = np.mean([seqeval_results[k]["f1"]        for k in keys])
    return float(prec), float(rec), float(f1)

# ---------------- Adapter paths ----------------
def adapter_pt_path(K:int, seg:int) -> str:
    p = os.path.join(ADAPTER_ROOT, f"win-{K}", f"from_wiki_segment_{seg}", "best_by_val", "adapter.pt")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Adapter not found: {p}")
    return p

# ---------------- Best-metric tracker (no checkpoints) ----------------
class ReportBestNoSave(TrainerCallback):
    """
    Tracks the best validation metric in memory (no disk saves), and when dev improves:
      - computes test metrics (and timing)
      - computes train/val/test mean cosine
      - stores dev macro metrics (requires dev predict)
    We finally expose a dict with the "best" snapshot metrics.
    """
    def __init__(self, trainer: Trainer, metric_name="eval_f1", greater_is_better=True,
                 tokenized_dataset=None, data_collator=None, batch_size=32):
        self.trainer = trainer
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.tokenized_dataset = tokenized_dataset
        self.data_collator = data_collator
        self.batch_size = batch_size

        self.best_val = None
        self.best_payload = None

    def _is_better(self, x, y):
        return (x > y) if self.greater_is_better else (x < y)

    def _seqeval_from_predict(self, preds_logits, labels):
        preds = np.argmax(preds_logits, axis=2)
        true_predictions = [[id2label[p] for (p, l) in zip(pred, lab) if l != -100] for pred, lab in zip(preds, labels)]
        true_labels      = [[id2label[l] for (p, l) in zip(pred, lab) if l != -100] for pred, lab in zip(preds, labels)]
        res = metric.compute(predictions=true_predictions, references=true_labels)
        return res

    def _cosines(self, model):
        from torch.utils.data import DataLoader
        train_dl = DataLoader(self.tokenized_dataset['train'], batch_size=self.batch_size, collate_fn=self.data_collator)
        eval_dl  = DataLoader(self.tokenized_dataset['validation'], batch_size=self.batch_size, collate_fn=self.data_collator)
        test_dl  = DataLoader(self.tokenized_dataset['test'], batch_size=self.batch_size, collate_fn=self.data_collator)
        mc_train = mean_cosine(model, train_dl)
        mc_val   = mean_cosine(model, eval_dl)
        mc_test  = mean_cosine(model, test_dl)
        return mc_train, mc_val, mc_test

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        if not metrics or self.metric_name not in metrics:
            return
        val = float(metrics[self.metric_name])
        if self.best_val is None or self._is_better(val, self.best_val):
            self.best_val = val

            # Dev macro (recompute from predictions for per-entity macro)
            preds_dev, labels_dev, _ = self.trainer.predict(self.tokenized_dataset['validation'])
            dev_overall = {
                "overall_precision": metrics.get("eval_precision"),
                "overall_recall": metrics.get("eval_recall"),
                "overall_f1": metrics.get("eval_f1"),
            }
            dev_seq = self._seqeval_from_predict(preds_dev, labels_dev)
            dev_macro_p, dev_macro_r, dev_macro_f1 = macro_scores(dev_seq)

            # Test metrics + timing at the moment of dev improvement
            t0 = time.time()
            preds_test, labels_test, _ = self.trainer.predict(self.tokenized_dataset['test'])
            test_time = time.time() - t0
            test_seq = self._seqeval_from_predict(preds_test, labels_test)
            test_overall = {
                "overall_precision": test_seq["overall_precision"],
                "overall_recall": test_seq["overall_recall"],
                "overall_f1": test_seq["overall_f1"],
                "overall_accuracy": test_seq.get("overall_accuracy", None),
            }
            test_macro_p, test_macro_r, test_macro_f1 = macro_scores(test_seq)

            # Cosines (train/val/test) at this "best" moment
            mc_train, mc_val, mc_test = self._cosines(model)

            self.best_payload = {
                "train_time_sec": None,  # filled by runner
                "test_time_sec": float(test_time),
                "dev_metrics": {
                    **dev_overall,
                    "macro_precision": dev_macro_p,
                    "macro_recall": dev_macro_r,
                    "macro_f1": dev_macro_f1
                },
                "test_metrics": {
                    **test_overall,
                    "macro_precision": test_macro_p,
                    "macro_recall": test_macro_r,
                    "macro_f1": test_macro_f1
                },
                "cosine": {
                    "train_mean": mc_train,
                    "validation_mean": mc_val,
                    "test_mean": mc_test
                },
                "best_step": state.global_step,
                "best_epoch": metrics.get("epoch", None),
                "best_val_metric": val
            }
            print(f"[Best@NoSave] New best {self.metric_name}={val:.6f} at step={state.global_step}, epoch={metrics.get('epoch')}")

    def get_best(self):
        return self.best_payload

# ---------------- Runner for a single case ----------------
def run_case(case_name: str, model: AdapterTokenClassifier, train_args: TrainingArguments,
             do_train: bool, tokenized_dataset, out_dir: str,
             param_stats: Dict[str, float]) -> Dict:

    os.makedirs(out_dir, exist_ok=True)
    trainer = Trainer(
        model=model, args=train_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Best-metric tracker (no saves)
    best_cb = ReportBestNoSave(
        trainer=trainer,
        metric_name="eval_f1",
        greater_is_better=True,
        tokenized_dataset=tokenized_dataset,
        data_collator=data_collator,
        batch_size=BATCH_SIZE
    )
    trainer.add_callback(best_cb)

    # Train
    if do_train:
        t0 = time.time()
        trainer.train()
        train_time = time.time() - t0
    else:
        train_time = 0.0

    best = best_cb.get_best()
    if best is None:
        # Fallback: run one eval/predict at end (shouldn't happen with evaluation_strategy="epoch")
        print("[Best@NoSave] WARNING: no eval events captured; computing once at end.")
        # Dev overall (from trainer.evaluate) + macro
        dev_eval = trainer.evaluate()
        preds_dev, labels_dev, _ = trainer.predict(tokenized_dataset['validation'])
        dev_seq = metric.compute(
            predictions=[[id2label[p] for (p,l) in zip(np.argmax(preds_dev,axis=2)[i], labels_dev[i]) if l!=-100]
                         for i in range(len(labels_dev))],
            references=[[id2label[l] for (p,l) in zip(np.argmax(preds_dev,axis=2)[i], labels_dev[i]) if l!=-100]
                         for i in range(len(labels_dev))]
        )
        dmp, dmr, dmf1 = macro_scores(dev_seq)
        # Test
        t0t = time.time()
        preds_test, labels_test, _ = trainer.predict(tokenized_dataset['test'])
        test_time = time.time() - t0t
        test_seq = metric.compute(
            predictions=[[id2label[p] for (p,l) in zip(np.argmax(preds_test,axis=2)[i], labels_test[i]) if l!=-100]
                         for i in range(len(labels_test))],
            references=[[id2label[l] for (p,l) in zip(np.argmax(preds_test,axis=2)[i], labels_test[i]) if l!=-100]
                         for i in range(len(labels_test))]
        )
        tmp, tmr, tmf1 = macro_scores(test_seq)
        # Cosines
        from torch.utils.data import DataLoader
        train_dl = DataLoader(tokenized_dataset['train'], batch_size=BATCH_SIZE, collate_fn=data_collator)
        eval_dl  = DataLoader(tokenized_dataset['validation'], batch_size=BATCH_SIZE, collate_fn=data_collator)
        test_dl  = DataLoader(tokenized_dataset['test'], batch_size=BATCH_SIZE, collate_fn=data_collator)
        mc_train = mean_cosine(model, train_dl)
        mc_val   = mean_cosine(model, eval_dl)
        mc_test  = mean_cosine(model, test_dl)

        best = {
            "train_time_sec": None,
            "test_time_sec": float(test_time),
            "dev_metrics": {
                "overall_precision": dev_eval.get("eval_precision"),
                "overall_recall": dev_eval.get("eval_recall"),
                "overall_f1": dev_eval.get("eval_f1"),
                "macro_precision": dmp, "macro_recall": dmr, "macro_f1": dmf1
            },
            "test_metrics": {
                "overall_precision": test_seq["overall_precision"],
                "overall_recall": test_seq["overall_recall"],
                "overall_f1": test_seq["overall_f1"],
                "overall_accuracy": test_seq.get("overall_accuracy", None),
                "macro_precision": tmp, "macro_recall": tmr, "macro_f1": tmf1
            },
            "cosine": {"train_mean": mc_train, "validation_mean": mc_val, "test_mean": mc_test},
            "best_step": None, "best_epoch": None, "best_val_metric": None
        }

    # fill train time
    best["train_time_sec"] = float(train_time)

    # persist JSON with best snapshot only
    out_json = {
        "case": case_name,
        "train_time_sec": best["train_time_sec"],
        "test_time_sec": best["test_time_sec"],
        "dev_metrics": best["dev_metrics"],
        "test_metrics": best["test_metrics"],
        "cosine": best["cosine"],
        "best_step": best.get("best_step"),
        "best_epoch": best.get("best_epoch"),
        "best_val_metric": best.get("best_val_metric"),
        "params": param_stats
    }
    # === Modified: round floats before writing JSON ===
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(round_floats(out_json, 2), f, indent=2)
    return out_json

# ---------------- Param accounting helpers ----------------
def component_params(model_obj: AdapterTokenClassifier) -> Dict[str, int]:
    backbone_params   = count_params(model_obj.backbone)
    adapter_params    = count_params(model_obj.fc1) + count_params(model_obj.fc2)
    classifier_params = count_params(model_obj.classifier)
    total_model_params = backbone_params + adapter_params + classifier_params
    return {
        "backbone_params": backbone_params,
        "adapter_params": adapter_params,
        "classifier_params": classifier_params,
        "backbone_adapter_params": backbone_params + adapter_params,
        "model_total_params": total_model_params,
    }

def component_trainable_params(model_obj: AdapterTokenClassifier) -> Dict[str, int]:
    tb = count_trainable_params(model_obj.backbone)
    ta = count_trainable_params(model_obj.fc1) + count_trainable_params(model_obj.fc2)
    tc = count_trainable_params(model_obj.classifier)
    return {"trainable_backbone": tb, "trainable_adapter": ta, "trainable_classifier": tc, "trainable_total": tb + ta + tc}

# ---------------- Main loop over 9 adapters × 3 cases ----------------
os.makedirs(OUT_ROOT, exist_ok=True)
global_summary = {}

for K in WINDOW_SIZES:
    for seg in WIKI_SEGS:
        tag = f"win-{K}/from_wiki_segment_{seg}"
        print(f"\n==== Running all cases for {tag} ====")
        adapter_path = adapter_pt_path(K, seg)

        def build_model(case: str) -> AdapterTokenClassifier:
            m = AdapterTokenClassifier(STUDENT_MODEL_NAME, num_labels=len(label_names), K=K, adapter_path=adapter_path)
            print(m)

            # flip trainables per case
            if case == "case1_eval_only":
                for p in m.backbone.parameters(): p.requires_grad = False
                for p in m.fc1.parameters(): p.requires_grad = False
                for p in m.fc2.parameters(): p.requires_grad = False
                for p in m.classifier.parameters(): p.requires_grad = True
            elif case == "case2_ft_adapter":
                for p in m.backbone.parameters(): p.requires_grad = False
                for p in m.fc1.parameters(): p.requires_grad = True
                for p in m.fc2.parameters(): p.requires_grad = True
                for p in m.classifier.parameters(): p.requires_grad = True
            elif case == "case3_ft_both":
                for p in m.backbone.parameters(): p.requires_grad = True
                for p in m.fc1.parameters(): p.requires_grad = True
                for p in m.fc2.parameters(): p.requires_grad = True
                for p in m.classifier.parameters(): p.requires_grad = True
            else:
                raise ValueError(case)

            # compute param stats AFTER setting requires_grad (for trainable breakdown)
            comp = component_params(m)
            trainable = component_trainable_params(m)

            pct_backbone_vs_bert = 100.0 * comp["backbone_params"] / TEACHER_TOTAL_PARAMS
            pct_backbone_adapter_vs_bert = 100.0 * comp["backbone_adapter_params"] / TEACHER_TOTAL_PARAMS
            pct_total_vs_bert = 100.0 * comp["model_total_params"] / TEACHER_TOTAL_PARAMS
            pct_trainable_of_model_before = 100.0 * trainable["trainable_total"] / comp["model_total_params"]

            m._param_stats = {
                # absolute counts
                "teacher_total_params": TEACHER_TOTAL_PARAMS,
                "backbone_params": comp["backbone_params"],
                "adapter_params": comp["adapter_params"],
                "backbone_adapter_params": comp["backbone_adapter_params"],
                "classifier_params": comp["classifier_params"],
                "model_total_params": comp["model_total_params"],
                # trainable (before FT)
                "trainable_backbone_before": trainable["trainable_backbone"],
                "trainable_adapter_before": trainable["trainable_adapter"],
                "trainable_classifier_before": trainable["trainable_classifier"],
                "model_trainable_before": trainable["trainable_total"],
                "pct_trainable_of_model_before": pct_trainable_of_model_before,
                # percentages vs BERT-base
                "pct_params_backbone_vs_bert": pct_backbone_vs_bert,
                "pct_params_backbone_adapter_vs_bert": pct_backbone_adapter_vs_bert,
                "pct_params_total_model_vs_bert": pct_total_vs_bert
            }

            m.config.id2label = id2label
            m.config.label2id = label2id
            return m.to(DEVICE)

        # ---- Common Trainer args (no saving; report best via callback) ----
        def args_for_case(out_dir, epochs, lr, wd):
            return TrainingArguments(
                output_dir=out_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=BATCH_SIZE,
                per_device_eval_batch_size=BATCH_SIZE,
                learning_rate=lr,
                weight_decay=wd,
                evaluation_strategy="epoch",
                save_strategy="no",               # <<< NO CHECKPOINTS
                load_best_model_at_end=False,      # <<< handled by our callback
                logging_steps=max(1, len(tokenized_dataset['train']) // BATCH_SIZE),
                seed=SEED,
                optim="adamw_torch",
                adam_beta1=ADAM_BETAS[0],
                adam_beta2=ADAM_BETAS[1],
                max_grad_norm=GRAD_CLIP,
                report_to=[]
            )

        # ---- Case 1: Eval only (train classifier only) ----
        case1_dir = os.path.join(OUT_ROOT, tag, "case1_eval_only")
        args1 = args_for_case(case1_dir, EPOCHS_CASE1, LR_CASE1, 0.0)
        model1 = build_model("case1_eval_only")
        res1 = run_case("case1_eval_only", model1, args1, do_train=True,
                        tokenized_dataset=tokenized_dataset, out_dir=case1_dir,
                        param_stats=model1._param_stats)

        # ---- Case 2: Fine-tune adapter only (+ head) ----
        case2_dir = os.path.join(OUT_ROOT, tag, "case2_ft_adapter")
        args2 = args_for_case(case2_dir, EPOCHS_CASE2, LR_CASE2, WEIGHT_DECAY)
        model2 = build_model("case2_ft_adapter")
        res2 = run_case("case2_ft_adapter", model2, args2, do_train=True,
                        tokenized_dataset=tokenized_dataset, out_dir=case2_dir,
                        param_stats=model2._param_stats)

        # ---- Case 3: Fine-tune adapter + compressed model (+ head) ----
        case3_dir = os.path.join(OUT_ROOT, tag, "case3_ft_both")
        args3 = args_for_case(case3_dir, EPOCHS_CASE3, LR_CASE3, WEIGHT_DECAY)
        model3 = build_model("case3_ft_both")
        res3 = run_case("case3_ft_both", model3, args3, do_train=True,
                        tokenized_dataset=tokenized_dataset, out_dir=case3_dir,
                        param_stats=model3._param_stats)

        # Collect per-adapter summary
        global_summary[tag] = {"case1": res1, "case2": res2, "case3": res3}

# ---- Aggregate mean cosines across 9 adapters (best snapshots) ----
def extract_cosines(results: Dict) -> Tuple[float,float,float]:
    return results["cosine"]["train_mean"], results["cosine"]["validation_mean"], results["cosine"]["test_mean"]

agg = {"case1": {"train": [], "val": [], "test": []},
       "case2": {"train": [], "val": [], "test": []},
       "case3": {"train": [], "val": [], "test": []}}
for tag, cases in global_summary.items():
    for cname in ["case1","case2","case3"]:
        tr, v, t = extract_cosines(cases[cname])
        agg[cname]["train"].append(tr); agg[cname]["val"].append(v); agg[cname]["test"].append(t)

agg_means = {
    "case1": {"train_mean_cosine": float(np.mean(agg["case1"]["train"])),
              "val_mean_cosine":   float(np.mean(agg["case1"]["val"])),
              "test_mean_cosine":  float(np.mean(agg["case1"]["test"]))},
    "case2": {"train_mean_cosine": float(np.mean(agg["case2"]["train"])),
              "val_mean_cosine":   float(np.mean(agg["case2"]["val"])),
              "test_mean_cosine":  float(np.mean(agg["case2"]["test"]))},
    "case3": {"train_mean_cosine": float(np.mean(agg["case3"]["train"])),
              "val_mean_cosine":   float(np.mean(agg["case3"]["val"])),
              "test_mean_cosine":  float(np.mean(agg["case3"]["test"]))},
}

os.makedirs(OUT_ROOT, exist_ok=True)
# === Modified: round floats before writing global JSON ===
with open(os.path.join(OUT_ROOT, "global_summary.json"), "w") as f:
    json.dump(round_floats({"adapters": global_summary, "aggregate_cosine_means": agg_means}, 2), f, indent=2)

# ---- Write combined CSV (best snapshots; split param fields) ----
csv_path = os.path.join(OUT_ROOT, "combined_results.csv")
fieldnames = [
    "window_K","wiki_segment","case",
    "train_time_sec","test_time_sec",
    "cosine_train","cosine_val","cosine_test",
    "dev_overall_precision","dev_overall_recall","dev_overall_f1",
    "dev_macro_precision","dev_macro_recall","dev_macro_f1",
    "test_overall_precision","test_overall_recall","test_overall_f1","test_overall_accuracy",
    "test_macro_precision","test_macro_recall","test_macro_f1",
    # param fields (split)
    "teacher_total_params",
    "backbone_params","adapter_params","backbone_adapter_params","classifier_params","model_total_params",
    "trainable_backbone_before","trainable_adapter_before","trainable_classifier_before","model_trainable_before",
    "pct_trainable_params_of_model_before",
    "pct_params_backbone_vs_bert","pct_params_backbone_adapter_vs_bert","pct_params_total_model_vs_bert",
]

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for tag, cases in global_summary.items():
        K_val = int(tag.split('/')[0].split('-')[-1])
        seg_val = int(tag.split('_')[-1])
        for case_name, res in cases.items():
            d = res["dev_metrics"]; t = res["test_metrics"]; c = res["cosine"]; p = res["params"]
            row = {
                "window_K": K_val,
                "wiki_segment": seg_val,
                "case": case_name,
                "train_time_sec": res["train_time_sec"],
                "test_time_sec": res["test_time_sec"],
                "cosine_train": c["train_mean"],
                "cosine_val": c["validation_mean"],
                "cosine_test": c["test_mean"],
                "dev_overall_precision": d["overall_precision"],
                "dev_overall_recall": d["overall_recall"],
                "dev_overall_f1": d["overall_f1"],
                "dev_macro_precision": d["macro_precision"],
                "dev_macro_recall": d["macro_recall"],
                "dev_macro_f1": d["macro_f1"],
                "test_overall_precision": t["overall_precision"],
                "test_overall_recall": t["overall_recall"],
                "test_overall_f1": t["overall_f1"],
                "test_overall_accuracy": t["overall_accuracy"],
                "test_macro_precision": t["macro_precision"],
                "test_macro_recall": t["macro_recall"],
                "test_macro_f1": t["macro_f1"],
                "teacher_total_params": p["teacher_total_params"],
                "backbone_params": p["backbone_params"],
                "adapter_params": p["adapter_params"],
                "backbone_adapter_params": p["backbone_adapter_params"],
                "classifier_params": p["classifier_params"],
                "model_total_params": p["model_total_params"],
                "trainable_backbone_before": p["trainable_backbone_before"],
                "trainable_adapter_before": p["trainable_adapter_before"],
                "trainable_classifier_before": p["trainable_classifier_before"],
                "model_trainable_before": p["model_trainable_before"],
                "pct_trainable_params_of_model_before": p["pct_trainable_of_model_before"],
                "pct_params_backbone_vs_bert": p["pct_params_backbone_vs_bert"],
                "pct_params_backbone_adapter_vs_bert": p["pct_params_backbone_adapter_vs_bert"],
                "pct_params_total_model_vs_bert": p["pct_params_total_model_vs_bert"],
            }
            # === Modified: round floats before writing CSV row ===
            writer.writerow(round_floats(row, 2))

print("\n=== Aggregate mean cosine across 9 adapters (best snapshots) ===")
print(json.dumps(agg_means, indent=2))
print(f"\nCombined CSV written to: {csv_path}")

