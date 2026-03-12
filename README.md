# autoresearch (Skills version)

A skill that runs machine learning and deep learning experiments end-to-end. Describe a task in plain English — the skill designs the pipeline, writes the code, executes it, tracks every trial, and delivers an actionable report.

## What it does

Given a request like *"train a sentiment classifier on this CSV"* or *"fine-tune DistilBERT and compare it to a logistic regression baseline"*, autoresearch will:

1. Inspect the data and state a brief plan
2. Write a complete, self-contained training script (`experiment/train.py`)
3. Set up an isolated `uv` virtual environment and run the script
4. Log every trial — including crashes and timeouts — to `experiment/results/experiment_log.tsv`
5. Write a report with results, a training curve, and concrete next-step recommendations

## Requirements

- Git (workspace must be a Git repository)
- [uv](https://github.com/astral-sh/uv) for Python environment management
- GPU recommended; falls back to CPU automatically

## Security

It is recommended to use [safe-chain](https://github.com/AikidoSec/safe-chain) to protect against supply chain attacks when running install scripts.

**Linux/macOS:**
```bash
curl -fsSL https://github.com/AikidoSec/safe-chain/releases/latest/download/install-safe-chain.sh | sh
```

**Windows (PowerShell):**
```powershell
iex (iwr "https://github.com/AikidoSec/safe-chain/releases/latest/download/install-safe-chain.ps1" -UseBasicParsing)
```

## Installation

```bash
bunx skills add SakaiSec/autoresearch
```

## Triggering the skill

The skill activates automatically on ML/training-related requests — no explicit invocation needed. Examples:

- "Train a classifier on `data.csv`"
- "Run a hyperparameter search over learning rates 1e-3, 1e-4, 1e-5"
- "Fine-tune DistilBERT on this dataset and report F1"
- "See if a transformer beats a random forest here"
- "Optimize this model — keep trying until val loss stops improving"

It covers classical ML (scikit-learn), deep learning (PyTorch), and HuggingFace fine-tuning.

## Execution modes

**Single-pass (default):** implement one well-reasoned pipeline, run it once, deliver a full report. Used for straightforward training tasks and explicit hyperparameter searches.

**Iterative search (Option B):** run a bounded sequence of trials, each testing one hypothesis. Only accepted trials (those that improve the primary metric) are committed. All trials are logged, including rejected and crashed ones. Requires an agreed trial budget and per-trial time limit before starting.

For complex or open-ended requests the skill asks once which mode you prefer before proceeding.

## Output layout

```
experiment/
├── train.py                         ← the complete training script
├── trials/                          ← iterative mode only
│   ├── trial-001/
│   │   ├── metrics.json
│   │   ├── run_config.json
│   │   ├── checkpoints/
│   │   └── logs/
│   └── ...
├── results/                         ← always present in both modes
│   ├── experiment_log.tsv           ← full trial log (TSV)
│   ├── report.md                    ← human-readable report
│   ├── training_curve.png           ← loss/metric plot
│   ├── metrics.json                 ← single-pass only
│   └── run_config.json              ← single-pass only
└── data/                            ← derived/preprocessed data (if applicable)
```

Every session works on a dedicated Git branch (`experiments/<YYYYMMDD>-<task>`). Accepted iterative trials are committed individually with metric deltas in the commit message.

## References

Detailed reference material lives in [`references/`](references/):

| File | Contents |
|---|---|
| [`experiment-log-schema.md`](references/experiment-log-schema.md) | Full `experiment_log.tsv` field definitions, lineage rules, log helper |
| [`frontier-management.md`](references/frontier-management.md) | Frontier admission criteria, eviction, size cap, TSV tracking |
| [`huggingface-finetuning.md`](references/huggingface-finetuning.md) | HuggingFace `Trainer` API patterns and configuration |

## Original Repository

This skill is inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).
