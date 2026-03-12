# Fine-tuning HuggingFace Models

Use the `Trainer` API when fine-tuning transformer models — it handles device placement, gradient
accumulation, evaluation loops, and checkpointing cleanly.

## Key patterns

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
