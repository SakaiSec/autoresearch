---
name: autoresearch
description: >
  Automatically design, implement, run, and analyze machine learning and deep learning experiments.
  Use this skill whenever the user wants to train a model, run a hyperparameter search, fine-tune a
  pretrained model, compare experiment results, or do anything that involves an ML training loop —
  even if they just describe the task in plain English like "train a classifier on this data" or
  "see if a transformer does better than a baseline here". Triggers on requests involving PyTorch,
  HuggingFace, model training, fine-tuning, loss curves, eval metrics, hyperparameter tuning, or
  running any kind of learning algorithm on a dataset. When in doubt, use this skill — it covers
  both classical approaches (scikit-learn) and deep learning.
---

# autoresearch

You help users run machine learning and deep learning experiments end-to-end. Given a plain-English
description of a task, you: understand the problem, write clean runnable code, execute it, track
results, and report findings — with minimal interruptions to the user.

---

## Understanding the task

When the user describes an experiment, extract:

1. **Problem type** — classification, regression, sequence labeling, generation, etc.
2. **Data** — what dataset or files are available, their format, rough size
3. **Model family** — do they have a preference? If not, pick something sensible:
   - Tabular / small data → scikit-learn baseline first, then a small MLP
   - Sequences / NLP → HuggingFace pretrained model (DistilBERT, etc.) or LSTM
   - Images → torchvision pretrained model (ResNet18, EfficientNet-B0) or simple CNN
   - General deep learning → PyTorch with sensible architecture
4. **Goal** — accuracy, F1, BLEU, MSE, perplexity, or whatever metric matters here
5. **Constraints** — time, hardware, dataset size limits

If you're missing something critical (especially the data), ask once briefly. For everything else,
make reasonable assumptions and proceed.

---

## Workflow

### 1. Inspect the data

Before writing any training code, understand the data:
- Read a sample (first ~20 rows for tabular, a few examples for text/images)
- Check class distribution for classification tasks
- Identify any obvious preprocessing needed (nulls, encoding, normalization)
- Note approximate size — this affects architecture and batch size choices

### 2. State a brief plan

In one short paragraph, say what you're going to do and why — model choice, data split,
key hyperparameters. This gives the user a chance to redirect before you write code.

### 3. Write the training script

Always write a complete, self-contained Python script saved to `experiment/train.py`. It must:

**Structure:**
- Parse args with `argparse` so hyperparameters are easy to override from the command line
- Have a `set_seed(seed)` function that seeds Python `random`, NumPy, and PyTorch (+ CUDA)
- Detect and use the best available device automatically (see Device Handling below)
- Save a checkpoint to `experiment/checkpoints/` at the best validation metric
- Write final metrics to `experiment/results/metrics.json`
- Save the exact args to `experiment/results/run_config.json`

**Logging:**
- Print loss/metrics every N steps with a simple `tqdm` progress bar
- Print a clean summary table at the end of training

**Code quality:**
- Scripts must be importable: use `if __name__ == "__main__":` guard
- Every function must be fully implemented — no stubs or `pass` placeholders

### 4. Run the experiment

Use `uv` for all Python execution and dependency management — never bare `python` or `pip`.

**Setting up the environment:**
```bash
# Create a virtual environment and install dependencies in one step
uv venv experiment/.venv
uv pip install --python experiment/.venv torch tqdm numpy scikit-learn
# Add more packages as needed, e.g.:
uv pip install --python experiment/.venv transformers datasets evaluate accelerate
```

**Running the script:**
```bash
uv run --python experiment/.venv python experiment/train.py
# or equivalently, after activating:
uv run python experiment/train.py
```

**Adding a missing package mid-run:**
```bash
uv pip install --python experiment/.venv <package-name>
```

Use `uv run` every time you execute Python — never call `python` or `python3` directly, and never use `pip install`.

Execute `uv run python experiment/train.py` and capture output. If it fails:
- Read the full traceback carefully
- Fix the root cause (don't just retry blind)
- Explain what you fixed in one sentence

If training is taking longer than expected with no visible progress, verify the script isn't stuck
(e.g., infinite data loop, deadlocked DataLoader worker).

### 5. Report results

After the run, write `experiment/results/report.md`:

```
# Experiment Report: [task name]

## Setup
- Model: ...
- Dataset: N train / M val examples
- Hardware: ...
- Training time: ...

## Results
| Metric | Value |
|--------|-------|

## Training curve (final epochs)
[last 5 epochs of train loss / val loss / key metric]

## Analysis
[2-4 sentences: what worked, what didn't, why]

## What to try next
1. [concrete suggestion with rationale]
2. [concrete suggestion with rationale]
3. [concrete suggestion with rationale]
```

The "What to try next" section should be concrete and actionable — not generic advice like
"try more data", but specific things like "add a cosine LR schedule" or "freeze encoder
layers 0–6 and only fine-tune the top 6".

---

## Hyperparameter search

When the user asks for a hyperparameter search (or after a baseline that leaves clear room for
improvement), run a small grid or random search:

- Default to ≤12 configs unless the user asks for more
- Run each config sequentially (or in subprocesses if the jobs are fast)
- Log every run to `experiment/results/experiment_log.csv` (see Experiment Tracking below)
- At the end, print a table ranked by validation metric and highlight the winner
- Optionally retrain the winner for the full number of epochs

Use `itertools.product` for grid search and `random.sample` over a param space for random search —
no extra libraries needed.

---

## Fine-tuning HuggingFace models

Use the `Trainer` API when fine-tuning transformer models — it handles device placement, gradient
accumulation, evaluation loops, and checkpointing cleanly.

Key patterns:
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import evaluate

tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = Dataset.from_pandas(df)
tokenized = dataset.map(
    lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=128),
    batched=True
)

metric = evaluate.load("f1")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return metric.compute(predictions=preds, references=labels, average="weighted")

args = TrainingArguments(
    output_dir="experiment/checkpoints",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_dir="experiment/logs",
    report_to="none",       # don't require W&B
)
trainer = Trainer(
    model=model, args=args,
    train_dataset=tokenized_train, eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
)
trainer.train()
```

**Model size guidance:**
- < 10K examples → DistilBERT (`distilbert-base-uncased`)
- < 100K examples → BERT-base (`bert-base-uncased`)
- Larger / multilingual → ask the user, or default to `roberta-base`

---

## Device handling

Always auto-detect the best device:

```python
import torch

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  Device: GPU — {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("  Device: Apple MPS")
    else:
        device = torch.device("cpu")
        print("  Device: CPU")
    return device
```

For raw PyTorch, call `.to(device)` on both model and input batches.
For HuggingFace Trainer, leave `no_cuda=False` (default) and it handles placement.

---

## Experiment tracking (lightweight)

No external tools required. Log every run to a local TSV so multiple runs are easy to compare
(TSV avoids delimiter conflicts with numeric values and model names):

```python
import csv, os
from datetime import datetime

def log_experiment(config: dict, metrics: dict, path="experiment/results/experiment_log.tsv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row = {"timestamp": datetime.now().isoformat(), **config, **metrics}
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys(), delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
```

Call this at the end of every training run. For hyperparameter searches, this naturally
builds a sortable table of all configs and their results.

---

## Common pitfalls — check for these

- Forgetting `.eval()` mode during validation (Dropout and BatchNorm behave differently)
- Not calling `optimizer.zero_grad()` before each backward pass
- Data leakage: always split *before* any fit-to-data preprocessing (e.g., `StandardScaler.fit`)
- Reporting only accuracy on imbalanced datasets — add F1 or ROC-AUC
- Leaving model on CPU when a GPU is available
- DataLoader `num_workers > 0` on Windows causing spawn issues

---

## Output file layout

```
experiment/
├── train.py                    ← the complete training script
├── checkpoints/
│   └── best_model/             ← saved weights at best val metric
├── results/
│   ├── metrics.json            ← {"accuracy": 0.87, "f1": 0.86, ...}
│   ├── run_config.json         ← exact hyperparameters used
│   ├── experiment_log.tsv      ← all runs (useful for hyperparameter search)
│   └── report.md               ← human-readable report with analysis
└── data/                       ← derived/preprocessed data (if applicable)
```

## Quality standards

- **Always use `uv`** for Python execution and package installs — never bare `python`, `python3`, or `pip`.
- **Do not hallucinate data.** If a data file doesn't exist or is unreadable, say so clearly.
- **Do not fake results.** Run the actual training. If hardware prevents it, say so explicitly.
- **Do not write placeholder code.** Every function must be complete.
- Scripts must be importable (use `if __name__ == "__main__":`) so users can extend them.
