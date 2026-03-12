# Frontier Management

The **frontier** is the small set of accepted trials that remain valid parents for future
exploration. It exists because the best trial on the primary metric is not always the only useful
starting point: a faster or smaller model that is close in quality may be worth branching from
when the best model is too expensive to iterate on quickly.

## Admission criteria

An accepted trial joins the frontier if it satisfies at least one of:

1. It holds the highest reported primary metric value among all accepted trials. When a tied trial
   is accepted (per the tie rule), it becomes the current best and satisfies this criterion.
2. It is within **2% relative degradation** of the current best on the primary metric *and*
   offers a **≥20% improvement** in at least one secondary metric (`runtime_sec`, `model_size_mb`,
   or `peak_vram_mb`) compared to every existing frontier member — i.e., it represents a tradeoff
   point not already covered.

## Eviction

Remove a trial from the frontier when a newer accepted trial dominates it: the newer trial is at
least as good on the primary metric *and* at least as good on every secondary dimension that
justified the dominated trial's membership. Evicted trials remain in the experiment log
permanently and retain their historical lineage; they are simply no longer valid parents.

## Size cap

The frontier must not exceed **4 members**. Before adding a fifth, re-evaluate all current members
for dominance and evict the weakest representative. In practice, one to three members is typical.

## Using the frontier

At the start of each new trial, choose the working base from the frontier. The default is the
current primary-metric best (criterion 1). Branch from a cost/quality tradeoff member only when
the experimental context specifically motivates it; record that member's `trial_id` as
`parent_trial_id`.

## Tracking membership in the log

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
