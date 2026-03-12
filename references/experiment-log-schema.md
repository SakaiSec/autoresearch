# Experiment Log Schema

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
| `artifacts_path` | Path to this trial's output directory; see Artifact directories below; empty for `frontier_eviction` rows |

## Artifact directories

- *Iterative runs:* each trial's outputs must be saved to `experiment/trials/<trial_id>/`
  (e.g., `experiment/trials/trial-003/`). Set `artifacts_path` to that path. This isolates each
  trial's outputs, makes individual trials easy to inspect and reproduce, and eliminates
  cross-trial file contamination.
- *Single-pass runs:* outputs go to `experiment/results/`. Set `artifacts_path=experiment/results/`
  unless the run creates a dedicated subdirectory, in which case use that path instead.

## Trial lineage rules

- Every trial must have a unique, zero-padded `trial_id` assigned sequentially (`trial-001`,
  `trial-002`, …).
- Every non-initial trial must record the `parent_trial_id` of the frontier member it was derived
  from.
- Only trials currently on the frontier may be used as `parent_trial_id` for new trials.
- Rejected, crashed, and timed-out trials remain in the log but are never on the frontier and
  must not be used as parents.
- Evicted trials remain in the log and retain their historical lineage links; they are no longer
  eligible as parents for new trials.

## Single-pass field conventions

Single-pass runs should still append a row to `experiment_log.tsv` so the file is a complete
execution record. Use these values for fields that are otherwise iterative-specific:

| Field | Value for single-pass runs |
|---|---|
| `trial_id` | `trial-001` (there is only one run) |
| `parent_trial_id` | empty |
| `event_type` | `trial` |
| `mode` | `single_pass` |
| `status` | `accepted` if the run completed; `crashed` or `timeout` otherwise |
| `accepted` | `true` if the run completed successfully; `false` otherwise |
| `on_frontier` | `false` (frontier is not used in single-pass mode) |
| `artifacts_path` | `experiment/results/` |

## Log helper

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
