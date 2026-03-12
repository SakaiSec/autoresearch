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

The workspace is always a Git repository. All experiment state management may rely on Git.

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

## Execution path: single-pass or iterative

Most requests are straightforward: write a script, run it once, report results. Proceed directly
to the workflow for those — including explicit, well-scoped hyperparameter tuning requests such as
"try learning rates 1e-3, 1e-4, 1e-5".

Before writing any code, check whether the request has any of these characteristics:

- **Multi-stage pipeline composition** — e.g., preprocessing → feature extraction → alignment →
  model training as distinct coupled stages, any of which could fail or need redesign
- **Implicit or open-ended iterative optimization** — language like "optimize", "align",
  "scan and train", "keep improving until…", or any framing that implies an unbounded search
- **Unclear stopping condition or evaluation budget** — no fixed epoch count, no explicit trial
  limit, no stated compute ceiling
- **Ontology mapping, label generation, or taxonomy alignment** — e.g., mapping raw events to
  MITRE ATT&CK tactics/techniques, aligning cluster labels to a schema, generating pseudo-labels
  iteratively
- **Potentially high compute cost or combinatorial growth** — large dataset × many model variants,
  or a search space that could silently expand to dozens of runs


If one or more criteria apply, stop and ask this question **once**, exactly as written, before
proceeding:

> **Before I begin:** this looks like it could go in two directions — I want to make sure I use
> your compute budget well.
>
> **Option A — Single pass:** I implement one well-reasoned pipeline end-to-end, run it, and
> deliver a full report with results and next-step recommendations. Clean, bounded, reproducible.
>
> **Option B — Iterative search:** I run multiple trials, each testing one hypothesis or
> configuration change. I keep only changes that improve the validation metric and revert the rest
> (though all trials are logged). We agree on a trial budget and a per-trial time limit upfront.
>
> Which would you prefer, and are there any compute or time constraints I should know about?

Do not ask this question for ordinary, well-scoped training tasks or explicit hyperparameter
searches (e.g., "train a sentiment classifier on this CSV", "try these three learning rates").
The default for ambiguous requests that trigger the gate but receive no explicit user preference
is **Option A** — a single, well-reasoned pass.

---

## Git branching

At the start of every experiment session, create and check out a dedicated branch:

```bash
git checkout -b experiments/YYYYMMDD-<task>
```

- `YYYYMMDD` is today's date in that exact format (e.g., `20260312`).
- `<task>` is a short lowercase slug derived from the user request
  (e.g., `image-classifier`, `iris-classifier`, `sentiment-finetune`).
- Branch off `main` (or the current default branch) unless the user specifies otherwise.

This branch is the working surface for the entire session. All commits, accepted trial states,
and final artifacts live here. Do not commit experiment work directly to `main`.

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
uv venv experiment/.venv
uv pip install --python experiment/.venv torch tqdm numpy scikit-learn
# Add more packages as needed, e.g.:
uv pip install --python experiment/.venv transformers datasets evaluate accelerate
```

**Running the script:**
```bash
uv run --python experiment/.venv python experiment/train.py
```

**Adding a missing package mid-run:**
```bash
uv pip install --python experiment/.venv <package-name>
```

Use `uv run` every time you execute Python — never call `python` or `python3` directly, and never
use `pip install`.

**Crash handling:** if a run fails, read the full traceback and attempt up to **three** targeted
fixes for obvious causes (wrong dtype, missing package, shape mismatch, etc.). If the run is still
failing after three attempts, mark the trial `crashed` in the experiment log with the error
summary and stop — do not loop indefinitely on a broken configuration.

**Wall-clock budget:** every run must have an implicit or explicit time ceiling. For single-pass
runs, flag to the user if training has produced no output after 10 minutes. For iterative trials,
enforce the per-trial time limit agreed with the user; a trial that exceeds its budget is treated
as a failure (`timeout`) — log it and move on rather than waiting for it to finish.

If training is taking longer than expected with no visible progress, verify the script isn't stuck
(e.g., infinite data loop, deadlocked DataLoader worker) before assuming it's just slow.

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

The "What to try next" section must be concrete and actionable — not generic advice like "try more
data", but specific things like "add a cosine LR schedule" or "freeze encoder layers 0–6 and only
fine-tune the top 6".

---

## Iterative trial discipline (Option B only)

When the user has chosen iterative search, execute it as a sequence of bounded, discrete trials.

### What a trial is

A **trial** is exactly:
- **One hypothesis or change set** — a single, clearly stated idea being tested (e.g., "add
  dropout 0.3 before the classifier head"). Do not bundle multiple independent changes into one
  trial; that makes causality impossible to attribute.
- **One execution run** — execute the trial configuration once. Do not silently re-run on failure
  unless it falls within the crash-handling budget (three attempts, then mark `crashed`).
- **One evaluation decision** — after the run completes (or fails), compare the primary metric
  against the current best and make a binary accept/reject call. See Metric semantics below.

### Accept / revert logic

In iterative mode, `HEAD` always points to the last accepted commit. Rejected, crashed, and
timed-out trials are **never committed**, so `HEAD` never advances past an accepted state.
Restore operations therefore always return the workspace to the last accepted state.

- **Accepted:** the primary metric improved (see tie rule below). Commit the working-state changes
  to the experiment branch — this advances `HEAD` to the new accepted state. Record the trial as
  `accepted` in the log and update the frontier (see Frontier management below).
- **Rejected:** the primary metric did not improve. The trial is not committed. Restore the
  workspace to the last accepted commit (`HEAD`) with:
  ```bash
  git reset --hard HEAD
  git clean -fd
  ```
  Record the trial as `rejected` in the log.
- **Crashed / timed out:** the trial is not committed. Restore the workspace to the last accepted
  commit (`HEAD`) with the same commands:
  ```bash
  git reset --hard HEAD
  git clean -fd
  ```
  Record the trial as `crashed` or `timeout` in the log.

The `git reset --hard HEAD && git clean -fd` sequence is required after every non-accepted trial.
`git reset --hard HEAD` restores all tracked files to the last accepted commit; `git clean -fd`
removes any untracked files and directories written during the trial. Together they guarantee a
clean slate before the next trial begins, with no artifacts from the failed attempt leaking
forward.

Git is used here for working-state control only. The experiment log and report artifacts are the
authoritative record of what was tried; they must remain complete regardless of what Git contains.
Reverted trials must remain fully visible in the log.

**Tie rule:** when a new trial matches the current best on the primary metric exactly, accept it
and it becomes the new canonical best (most-recently-accepted wins). Both the prior best and the
new trial are eligible for frontier membership under the cost/quality tradeoff criterion.

### Frontier management

The **frontier** is the small set of accepted trials that remain valid parents for future
exploration. It exists because the best trial on the primary metric is not always the only useful
starting point: a faster or smaller model that is close in quality may be worth branching from
when the best model is too expensive to iterate on quickly.

**Admission criteria.** An accepted trial joins the frontier if it satisfies at least one of:

1. It is the current best on the primary metric (the most-recently-accepted trial among any ties).
2. It is within **2% relative degradation** of the current best on the primary metric *and*
   offers a **≥20% improvement** in at least one secondary metric (`runtime_sec`, `model_size_mb`,
   or `peak_vram_mb`) compared to every existing frontier member — i.e., it represents a tradeoff
   point not already covered.

**Eviction.** Remove a trial from the frontier when a newer accepted trial dominates it: the newer
trial is at least as good on the primary metric *and* at least as good on every secondary
dimension that justified the dominated trial's membership. Evicted trials remain in the experiment
log permanently and retain their historical lineage; they are simply no longer valid parents.

**Size cap.** The frontier must not exceed **4 members**. Before adding a fifth, re-evaluate all
current members for dominance and evict the weakest representative. In practice, one to three
members is typical.

**Using the frontier.** At the start of each new trial, choose the working base from the frontier.
The default is the current primary-metric best (criterion 1). Branch from a cost/quality tradeoff
member only when the experimental context specifically motivates it; record that member's
`trial_id` as `parent_trial_id`.

Frontier membership is tracked via the `on_frontier` field in the experiment log. Because TSV
rows are append-only, record an eviction by appending a dedicated bookkeeping row:

| Field | Value |
|---|---|
| `event_type` | `frontier_eviction` |
| `trial_id` | `trial_id` of the evicted trial |
| `on_frontier` | `false` |
| `revert_reason` | brief reason for eviction (e.g., `dominated by trial-005 on val_f1 and runtime_sec`) |
| all other fields | empty |

Setting `on_frontier=false` on eviction rows (rather than leaving it blank) ensures a log reader
can reconstruct the current frontier state deterministically by scanning `event_type=trial` rows
for their most recent `on_frontier` value, then applying `frontier_eviction` rows in order.

### Git commits for accepted trials

Each accepted trial must produce a commit on the experiment branch:

```bash
git add experiment/
git commit -m "trial-001: <short description> [<metric>=<value> +<delta>]"
```

Example: `trial-003: add cosine LR schedule [val_f1=0.847 +0.012]`

Include the zero-padded trial identifier, a one-line description of the change, and the primary
metric value with its delta from the previous best. Do not commit rejected, crashed, or timed-out
trials — only accepted trials advance `HEAD`.

### Per-trial time limit

Before starting iterative search, confirm a wall-clock budget per trial with the user
(e.g., "5 minutes per trial, up to 10 trials"). A trial that exceeds its limit is `timeout`:
restore the working state and log it as a failure. Treat a repeatedly timing-out configuration
as a redesign candidate, not something to retry with a larger budget without user confirmation.

### Stopping condition

Iterative search must have an explicit stopping condition agreed before it starts: a maximum
number of trials, a target metric threshold, or a total wall-clock budget. Do not run open-ended
loops. When the stopping condition is reached, write the final report and stop.

---

## Metric semantics

**Primary metric** is the single metric used to determine trial acceptance. It must be agreed
before iterative search begins (e.g., `val_f1`, `val_loss`, `val_accuracy`). Trial acceptance
decisions must depend only on the primary metric — not on secondary metrics.

**Secondary metrics** are optional diagnostics recorded for analysis but not used in acceptance
decisions. Useful secondary metrics include:

| Metric | Purpose |
|---|---|
| `runtime_sec` | Flags unexpectedly slow trials |
| `peak_vram_mb` | Tracks GPU memory pressure |
| `model_size_mb` | Monitors parameter growth |
| `train_loss` | Sanity-check for overfitting |
| `val_loss` | Useful when primary metric is accuracy-based |

---

## Experiment log schema

Every trial — accepted, rejected, crashed, and timed-out — must be appended to
`experiment/results/experiment_log.tsv`. Omitting any trial is not permitted.

Required fields:

| Field | Description |
|---|---|
| `trial_id` | Zero-padded sequential identifier (e.g., `trial-001`, `trial-002`) |
| `parent_trial_id` | `trial_id` of the frontier member this trial was derived from; empty for the initial baseline |
| `event_type` | `trial` for normal trial rows; `frontier_eviction` for eviction bookkeeping rows |
| `mode` | `single_pass` or `iterative` |
| `hypothesis` | One-sentence statement of what this trial is testing |
| `change_summary` | Brief description of what was changed relative to the parent |
| `runtime_sec` | Measured wall-clock duration of the run |
| `status` | `accepted`, `rejected`, `crashed`, or `timeout`; empty for bookkeeping rows |
| `primary_metric` | Name of the acceptance metric (e.g., `val_f1`) |
| `primary_metric_value` | Measured value; empty if the run did not complete |
| `secondary_metrics` | JSON-encoded dict of additional metrics; `{}` if none |
| `best_so_far` | Value of the primary metric in the current accepted working state after this trial |
| `accepted` | `true` or `false`; empty for `frontier_eviction` rows |
| `on_frontier` | `true` or `false`; must be `false` on `frontier_eviction` rows — never left blank |
| `revert_reason` | Why the trial was not accepted, or why it was evicted from the frontier; empty for accepted trial rows |
| `artifacts_path` | `experiment/trials/<trial_id>/` for `trial` rows; empty for `frontier_eviction` rows |

**Artifact directories.** Each trial's outputs must be saved to `experiment/trials/<trial_id>/`
(e.g., `experiment/trials/trial-003/`). Store metrics, logs, checkpoints, and any debug artifacts
there. This isolates each trial's outputs, makes individual trials easy to inspect and reproduce,
and eliminates cross-trial file contamination.

**Trial lineage rules:**
- Every trial must have a unique, zero-padded `trial_id` assigned sequentially (`trial-001`,
  `trial-002`, …).
- Every non-initial trial must record the `parent_trial_id` of the frontier member it was derived
  from.
- Only trials currently on the frontier may be used as `parent_trial_id` for new trials.
- Rejected, crashed, and timed-out trials remain in the log but are never on the frontier and
  must not be used as parents.
- Evicted trials remain in the log and retain their historical lineage links; they are no longer
  eligible as parents for new trials.

Log helper:

```python
import csv, json, os
from datetime import datetime

def log_trial(record: dict, path="experiment/results/experiment_log.tsv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    record = {"timestamp": datetime.now().isoformat(), **record}
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=record.keys(), delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerow(record)
```

Encode `secondary_metrics` as a JSON string before passing to the logger:
`record["secondary_metrics"] = json.dumps({"peak_vram_mb": 4200, "model_size_mb": 438})`.

---

## Hyperparameter search

When the user explicitly requests a hyperparameter search over a defined grid or list (e.g.,
"try learning rates 1e-3, 1e-4, 1e-5"), treat it as a single-pass execution: enumerate the
configs, run them sequentially, log each to `experiment_log.tsv`, print a ranked comparison
table, and identify the winner. No confirmation prompt is required.

Default to ≤12 configs unless the user asks for more. Use `itertools.product` for grid search
and `random.sample` over a param space for random search — no extra libraries needed.

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
├── train.py                         ← the complete training script
├── trials/
│   ├── trial-001/                   ← per-trial artifact directory
│   │   ├── metrics.json
│   │   ├── run_config.json
│   │   ├── checkpoints/
│   │   └── logs/
│   ├── trial-002/
│   └── ...
├── results/
│   ├── experiment_log.tsv           ← all trials; full schema including lineage fields
│   └── report.md                    ← human-readable report with analysis
└── data/                            ← derived/preprocessed data (if applicable)
```

Single-pass runs may write directly to `experiment/results/` and `experiment/checkpoints/`
without the `trials/` subdirectory structure.

---

## Quality standards

- **Always use `uv`** for Python execution and package installs — never bare `python`, `python3`, or `pip`.
- **Do not hallucinate data.** If a data file doesn't exist or is unreadable, say so clearly.
- **Do not fake results.** Run the actual training. If hardware prevents it, say so explicitly.
- **Do not write placeholder code.** Every function must be complete.
- **Do not run indefinitely.** Every execution path must be bounded by an explicit stopping
  condition, trial budget, or wall-clock limit agreed with the user before execution begins.
- Scripts must be importable (use `if __name__ == "__main__":`) so users can extend them.
