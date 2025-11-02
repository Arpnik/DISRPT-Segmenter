# 🧠 EDU Segmenter – BERT Fine-Tuning with LoRA

This repository provides code for **fine-tuning transformer models (like BERT/DistilBERT)** for **EDU segmentation** tasks, 
with support for **LoRA parameter-efficient training**, **early stopping**, and **Weights & Biases (W&B)** experiment tracking.

---

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.10+**
- **Git**
- **pip** (Python package manager)

Optional but recommended:
- **GPU with CUDA** for faster training
- A [Weights & Biases (W&B)](https://wandb.ai/site) account for run tracking
- [`uv`](https://github.com/astral-sh/uv) is a fast Python package and environment manager.
    ```bash
  
    pip install uv
    ````
---

## ⚙️ Steps to Follow

### 1. Download/Clone the repository

```bash 

git clone https://github.com/Arpnik/DISRPT-Segmenter.git
```


### 2. Create and activate a new environment

```bash

uv venv .venv
source .venv/bin/activate   # macOS / Linux
```

#### OR
```bash

.venv\Scripts\activate      # Windows
````

#### Optional
Set this parameter to ignore the tokenizer warning
```bash
export TOKENIZERS_PARALLELISM="true"
````

### 3. Install the Project Dependencies
Make sure you run these commands from from folder with pyproject.toml file
```bash

uv pip install -e .
````

### 4. Dataset download and verification
To download and validate the datasets, simply run:
```bash
python -m com.disrpt.segmenter.dataset_prep
```
This will:
- Download all defined corpora into the dataset/ directory.
- Load each dataset (train.conllu, dev.conllu, test.conllu) automatically.
- Display detailed statistics for EDU segmentation (tokens, EDU counts, and balance).
- Confirm token and label shapes match.

### 5. Fine-tune the model

There are 2 segmenters defined which use a transformer model (DistilBERT or BERT) with LoRA applied to the self-attention mechanism. The fundamental difference lies in the classification head used to predict the EDU boundaries.

1. **Segmenter_tuning.py**: Standard Linear Head (Hugging Face default for `AutoModelForTokenClassification`)
2. **fine_tuning.py**: Custom Multi-Layer Perceptron (MLP) Head


#### **Model Architecture Comparison**

| Feature | Segmenter_tuning.py | fine_tuning.py |
|---------|-------------------|----------------|
| Base Model | DistilBERT/BERT | DistilBERT/BERT |
| LoRA Adaptation | ✓ (query, value) | ✓ (query, value) |
| Classification Head | Single Linear Layer | Multi-Layer Perceptron |
| Head Architecture | `[768 → 2]` | `[768 → 256 → 128 → 2]` |
| Dropout | None | 0.3 (configurable) |
| Activation | None | GELU (configurable) |

---
#### **Running the Fine-tuning Scripts**

##### **Option 1: Segmenter_tuning.py (Linear Head)**

**Basic usage:**
```bash
python segmenter_tuning.py
```

**With custom parameters:**
```bash
python segmenter_tuning.py \
  --model_name distilbert-base-uncased \
  --output_dir ./output/edu_segmenter \
  --epochs 10 \
  --batch_size 16 \
  --learning_rate 3e-4 \
  --save_every_n_epochs 2 \
  --early_stopping_patience 3 \
  --use_wandb \
  --wandb_project edu-segmentation \
  --wandb_run_name distilbert-linear-head
```

**Default Parameters:**
- `--model_name`: `distilbert-base-uncased`
- `--output_dir`: `./output/edu_segmenter`
- `--epochs`: `10`
- `--batch_size`: `16`
- `--learning_rate`: `3e-4`
- `--save_every_n_epochs`: `2`
- `--early_stopping_patience`: `3`
- `--use_wandb`: `False` (flag to enable)
- `--wandb_project`: `edu-segmentation`
- `--wandb_run_name`: Auto-generated (model name + lr + epochs)

**LoRA Configuration (fixed):**
- LoRA rank (`r`): `16`
- LoRA alpha: `32`
- LoRA dropout: `0.1`
- Target modules: `["q_lin", "v_lin"]` (DistilBERT) or `["query", "value"]` (BERT)

---

##### **Option 2: fine_tuning.py (MLP Head)**

**Basic usage:**
```bash
python fine_tuning.py
```

**With custom parameters:**
```bash
python fine_tuning.py \
  --model_name distilbert-base-uncased \
  --output_dir ./output-2/edu_segmenter \
  --epochs 10 \
  --batch_size 16 \
  --learning_rate 3e-4 \
  --mlp_dims 256 128 \
  --mlp_dropout 0.3 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.1 \
  --save_every_n_epochs 2 \
  --early_stopping_patience 3 \
  --use_wandb \
  --wandb_project edu-segmentation \
  --wandb_run_name distilbert-mlp-head
```

**Default Parameters:**
- `--model_name`: `distilbert-base-uncased`
- `--output_dir`: `./output-2/edu_segmenter`
- `--epochs`: `10`
- `--batch_size`: `16`
- `--learning_rate`: `3e-4`
- `--save_every_n_epochs`: `2`
- `--early_stopping_patience`: `3`
- `--use_wandb`: `False` (flag to enable)
- `--wandb_project`: `edu-segmentation`
- `--wandb_run_name`: Auto-generated

**MLP Configuration:**
- `--mlp_dims`: `[256, 128]` (hidden layer dimensions)
- `--mlp_dropout`: `0.3`
- MLP activation: `GELU` (fixed)

**LoRA Configuration:**
- `--lora_r`: `16` (LoRA rank)
- `--lora_alpha`: `32`
- `--lora_dropout`: `0.1`
- Target modules: `["q_lin", "v_lin"]` (DistilBERT) or `["query", "value"]` (BERT)

---

#### **Available Model Options**

Both scripts support any HuggingFace BERT-based model:
- `distilbert-base-uncased` (default, faster)
- `bert-base-uncased`
- `bert-base-cased`
- `roberta-base`
- Any compatible model from HuggingFace Hub

---

#### **Weights & Biases (W&B) Integration**

Both scripts support optional W&B logging for experiment tracking:
```bash
# Enable W&B logging
python fine_tuning.py --use_wandb

# Custom W&B project and run name
python fine_tuning.py \
  --use_wandb \
  --wandb_project my-edu-project \
  --wandb_run_name experiment-001
```

**W&B logs include:**
- Training/validation metrics (accuracy, precision, recall, F1)
- Per-class metrics (for both EDU Continue and EDU Start)
- Loss curves
- Learning rate schedules
- Model architecture
- Confusion matrix
- Final results table
- Model artifacts (best checkpoint)

---

#### **Output Structure**

After training, the output directory will contain:
```
output/edu_segmenter/
├── best_model/              # Best model checkpoint
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   ├── config.json
│   ├── tokenizer_config.json
│   └── ...
├── logs/                    # TensorBoard logs
├── checkpoint-XXX/          # Periodic checkpoints
└── ...
```

---

#### **Training Pipeline Steps**

Both scripts execute the following pipeline:

1. **Download Dataset**: Automatically downloads GUM corpus
2. **Load Datasets**: Tokenizes and prepares train/dev/test splits
3. **Initialize Model**: Loads pre-trained model + applies LoRA
4. **Train Model**: Fine-tunes with early stopping
5. **Evaluate Test Set**: Final evaluation on held-out test data

---

#### **Hardware Requirements**

- **Minimum**: 8GB GPU VRAM (for DistilBERT)
- **Recommended**: 16GB+ GPU VRAM (for BERT-base)
- **CPU**: Training possible but significantly slower
- **Storage**: ~2GB for models and datasets

---

#### **When to Use Which Script?**

| Use Case | Recommended Script |
|----------|-------------------|
| Faster training, fewer parameters | segmenter_tuning.py |
| Better accuracy, more capacity | fine_tuning.py |
| Limited GPU memory | segmenter_tuning.py |
| Maximum performance | fine_tuning.py |
| Quick baseline | segmenter_tuning.py |
| Production deployment | fine_tuning.py |


#### **Example Training Sessions**

**Quick experiment (DistilBERT + Linear Head):**
```bash
python segmenter_tuning.py --epochs 5 --batch_size 32
```

**Full training with W&B (BERT + MLP Head):**
```bash
python fine_tuning.py \
  --model_name bert-base-uncased \
  --epochs 15 \
  --batch_size 16 \
  --learning_rate 2e-4 \
  --mlp_dims 512 256 128 \
  --use_wandb \
  --wandb_project edu-segmentation-final
```

**Resume training with different learning rate:**
```bash
python fine_tuning.py \
  --output_dir ./output-2/edu_segmenter \
  --learning_rate 1e-4 \
  --epochs 5
```