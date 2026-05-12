# Alignment Adapter to Improve the Performance of Compressed Models

This repository contains the implementation of **Alignment Adapter (AlAd)**, a lightweight sliding-window adapter designed to improve the downstream performance of compressed BERT-family models. AlAd aligns the token-level representations of a compressed model with those of a larger reference model and can be used either as a frozen-backbone plug-and-play module or jointly fine-tuned with the compressed model.

**Paper:** *Alignment Adapter to Improve the Performance of Compressed Models*  
**Authors:** Rohit Raj Rai, Abhishek Dhaka, Amit Awekar  
**Repository:** <https://github.com/rohitrai11/Alignment-Adapter-to-Improve-the-Performance-of-Compressed-Deep-Learning-Models>

---

## 1. Overview

Deep learning models are commonly compressed before deployment in resource-constrained environments. However, compression often reduces downstream task performance. AlAd addresses this by learning a lightweight transformation from the compressed model's token embeddings to the representation space of a larger teacher model.

For an input token sequence, the compressed model produces token embeddings. AlAd takes each token embedding together with a small local context window and maps it to the teacher-model embedding dimension. This helps restore local contextual information lost during compression while adding only a small number of parameters.

The repository supports experiments across three token-level NLP tasks:

| Task | Dataset used in the paper | Main evaluation metrics |
|---|---|---|
| POS tagging | German Universal Dependencies | Accuracy, overall F1 |
| NER | MultiCoNER v2 | Overall F1, macro F1 |
| EQA | SQuAD 2.0 | Exact Match, F1 |

The compressed models considered are:

| Model | Description |
|---|---|
| ASC | Application-specific compressed BERT model |
| BERT-Mini | Compact BERT model with smaller hidden dimension |
| BERT-Tiny | Highly compressed BERT model |

BERT-base is used as the large reference model for representation alignment.

---

## 2. Method Summary

AlAd is trained in three stages.

### Stage 1: Task-independent AlAd pretraining

The compressed model is frozen. AlAd is trained to map compressed-model token embeddings to BERT-base token embeddings using a general text corpus, such as English Wikipedia.

### Stage 2: Task-specific continual alignment

The compressed model remains frozen. AlAd is further trained on the task-specific dataset without using the task labels for final prediction. This adapts the adapter to the distribution of the target task.

### Stage 3: Task-specific fine-tuning

AlAd is attached to the compressed model and a task-specific prediction head is trained. Two modes are supported:

1. **Frozen compressed model:** only AlAd and the task head are trained.
2. **Joint fine-tuning:** AlAd, the compressed model, and the task head are trained together.

For EQA, adapter-based fine-tuning is implemented using multiple task-specific scripts rather than a single unified fine-tuning script.

---

## 3. Supported Tasks and Models

The same experimental structure is followed for **POS**, **NER**, and **EQA** tasks using the following compressed models:

- ASC
- BERT-Mini
- BERT-Tiny

For each task-model pair, the workflow is:

```text
1. Task-independent AlAd pretraining
2. Task-specific continual AlAd pretraining
3. Task-specific fine-tuning with adapters
   ├── frozen compressed model setting
   └── jointly fine-tuned compressed model setting
```

Window sizes used for AlAd:

```text
W1, W3, W5
```

where `W1` uses only the current token embedding, `W3` uses one neighboring token on each side, and `W5` uses two neighboring tokens on each side.

---

## 4. Repository Structure

A recommended organization is shown below. The exact paths may vary depending on how the scripts are placed in the repository.

```text
.
├── README.md
├── requirements.txt
│
├── NER/
│   ├── ASC/
│   │   ├── pretrain-asc-wall-clock.py
│   │   ├── continual-pretraining.py
│   │   └── fine_tune_asc_with_adapter_all_cases.py
│   │
│   ├── BERT-Mini/
│   │   ├── pretrain-bert-mini-wall-clock.py
│   │   ├── continual-pretraining.py
│   │   └── fine_tune_bert_mini_with_adapter_all_cases.py
│   │
│   └── BERT-Tiny/
│       ├── pretrain-bert-tiny-wall-clock.py
│       ├── continual-pretraining.py
│       └── fine_tune_bert_tiny_with_adapter_all_cases.py
│
├── POS/
│   ├── ASC/
│   ├── BERT-Mini/
│   └── BERT-Tiny/
│
├── EQA/
│   ├── ASC/
│   ├── BERT-Mini/
│   └── BERT-Tiny/
│
├── data/
│   ├── pos/
│   ├── ner/
│   └── eqa/
│
├── checkpoints/
│   ├── pretraining/
│   ├── continual_pretraining/
│   └── fine_tuning/
│
└── results/
    ├── pos/
    ├── ner/
    └── eqa/
```

For the **NER + BERT-Mini** setting, the key scripts are:

```text
pretrain-bert-mini-wall-clock.py
continual-pretraining.py
fine_tune_bert_mini_with_adapter_all_cases.py
```

The file `fine_tune_bert_mini_with_adapter_all_cases.py` includes both frozen-backbone and jointly fine-tuned settings.

---

## 5. Environment Setup

Create a clean Python environment:

```bash
conda create -n alad python=3.10 -y
conda activate alad
python -m pip install --upgrade pip
```

Install dependencies:

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements.txt
```

If installation fails because `requirements.txt` contains a machine-specific local entry such as `conda-pack @ file:///...`, create a cleaned requirement file:

```bash
grep -v "conda-pack @ file:" requirements.txt > requirements_clean.txt
pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements_clean.txt
```

For CPU-only installation, install the CPU version of PyTorch first and then install the remaining dependencies:

```bash
pip install torch torchvision torchaudio
pip install -r requirements_clean.txt
```

---

## 6. Running Experiments

The following commands show the expected workflow. Modify file paths, dataset paths, checkpoint paths, GPU IDs, and hyperparameters according to the script arguments used in each folder.

### 6.1 NER with BERT-Mini

Move to the BERT-Mini NER experiment directory:

```bash
cd NER/BERT-Mini
```

#### Step 1: Task-independent pretraining

```bash
python pretrain-bert-mini-wall-clock.py
```

This stage trains AlAd using a general corpus while the compressed model remains frozen.

#### Step 2: Continual pretraining on the NER dataset

```bash
python continual-pretraining.py
```

This stage adapts AlAd to the NER dataset distribution using representation alignment.

#### Step 3: Fine-tuning for NER

```bash
python fine_tune_bert_mini_with_adapter_all_cases.py
```

This script includes both cases:

- frozen compressed model + trainable AlAd
- jointly fine-tuned compressed model + trainable AlAd

Run the script separately for different window sizes.

Example window-size convention:

```bash
python fine_tune_bert_mini_with_adapter_all_cases.py --window_size 1
python fine_tune_bert_mini_with_adapter_all_cases.py --window_size 3
python fine_tune_bert_mini_with_adapter_all_cases.py --window_size 5
```

### 6.2 POS experiments

The POS task follows the same three-stage structure for ASC, BERT-Mini, and BERT-Tiny:

```text
1. pretraining script
2. continual-pretraining.py
3. fine-tuning script for frozen and joint settings
```

Example:

```bash
cd POS/BERT-Mini
python pretrain-bert-mini-wall-clock.py
python continual-pretraining.py
python fine_tune_bert_mini_with_adapter_all_cases.py
```

### 6.3 EQA experiments

For EQA, the same overall idea is followed, but adapter fine-tuning is implemented using multiple scripts rather than one combined fine-tuning script. Run the EQA scripts in the order specified inside the corresponding model directory.

Recommended workflow:

```text
1. Run task-independent AlAd pretraining.
2. Run continual pretraining on SQuAD 2.0.
3. Run the EQA frozen-backbone fine-tuning script, if available.
4. Run the EQA joint fine-tuning script, if available.
5. Evaluate Exact Match and F1.
```

---

## 7. Outputs

Each experiment should save outputs in a structured directory such as:

```text
results/<task>/<model>/<mode>/W<window_size>/
```

Recommended files to save:

```text
metrics.json
training_log.txt
model_config.json
adapter_checkpoint.pt
final_model_checkpoint.pt
predictions.json
```

For representation-level analysis, save token-level cosine similarity results, for example:

```text
cosine_similarity_results.json
```

---

## 8. Evaluation

Use task-specific evaluation metrics:

| Task | Metrics |
|---|---|
| POS | Accuracy, overall F1 |
| NER | Overall F1, macro F1 |
| EQA | Exact Match, F1 |

The paper also reports mean token-level embedding cosine similarity between compressed-model representations after AlAd projection and the fine-tuned BERT-base representations. This is useful for validating whether AlAd improves representation alignment.

---


## 9. Common Issues

### CUDA or PyTorch installation error

The provided environment uses PyTorch with CUDA 12.1. If your system uses another CUDA version, install the PyTorch build compatible with your machine from the official PyTorch installation instructions.

### `conda-pack @ file:///...` installation error

This line is machine-specific and may not install on another system. Remove it from `requirements.txt` or use `requirements_clean.txt` as shown in the setup section.

### Out-of-memory error

Try reducing:

- batch size
- maximum sequence length
- number of workers
- window size

For EQA, memory usage is usually higher than POS and NER because span prediction requires handling longer contexts.

---

## 10. Contact

For questions, please contact:

**Rohit Raj Rai**  
Indian Institute of Technology Guwahati  
Email: rohitraj@iitg.ac.in
