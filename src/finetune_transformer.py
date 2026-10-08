"""
Fine-tune a pretrained Transformer (e.g. AfroXLMR) for Kirundi news topic
classification on the deduplicated KIRNEWS splits made by data_prep.py.

Run on a GPU (Kaggle / Colab). Example:
    python src/finetune_transformer.py --model Davlan/afro-xlmr-base \
        --max_len 256 --weighted_loss --seed 42 --name E3_afroxlmr_weighted

Smoke test (1 minute, checks that everything runs):
    python src/finetune_transformer.py --model Davlan/afro-xlmr-base --smoke

Key ideas (be ready to explain these in the viva):
* Transfer learning: the model was pre-trained with masked language modelling
  on large text corpora; we add a small classification head (linear layer on
  top of the first token's hidden state) and train everything end to end.
* Class imbalance: some classes have <20 training articles, so optionally we
  weight the cross-entropy loss by (N / (K * n_c)) ** 0.5 (a damped inverse
  frequency) so rare classes count more without destabilising training.
* Model selection: the checkpoint with the best VALIDATION macro-F1 is kept;
  the TEST set is only used once, at the end.
"""
import argparse
import json
import math
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from torch import nn
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, EarlyStoppingCallback,
                          Trainer, TrainingArguments, set_seed)

CLASS_NAMES = ["politics", "sport", "economy", "health", "entertainment",
               "history", "technology", "culture", "religion",
               "environment", "education", "relationship"]


class TextDataset(torch.utils.data.Dataset):
    """Tokenises each article once. Truncation keeps the FIRST max_len tokens
    (news articles put the key information at the start)."""

    def __init__(self, texts, labels, tokenizer, max_len):
        self.enc = tokenizer(list(texts), truncation=True, max_length=max_len)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = int(self.labels[i])
        return item


class WeightedTrainer(Trainer):
    """Trainer with an optional class-weighted cross-entropy loss."""

    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        w = None if self.class_weights is None else self.class_weights.to(outputs.logits.device)
        loss = nn.CrossEntropyLoss(weight=w)(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro", zero_division=0)}


def main(a):
    set_seed(a.seed)
    random.seed(a.seed)
    data, out = Path(a.data_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tr, va, te = (pd.read_csv(data / f"{a.version}_{s}.csv") for s in ("train", "val", "test"))
    if a.smoke:
        tr, va, te = tr.sample(96, random_state=0), va.head(48), te.head(48)
        a.epochs = 1
    name = a.name or f"{a.model.split('/')[-1]}_len{a.max_len}_s{a.seed}"

    tok = AutoTokenizer.from_pretrained(a.model)
    ds_tr, ds_va, ds_te = (TextDataset(d.text, d.y, tok, a.max_len) for d in (tr, va, te))

    model = AutoModelForSequenceClassification.from_pretrained(
        a.model, num_labels=len(CLASS_NAMES),
        id2label=dict(enumerate(CLASS_NAMES)),
        label2id={c: i for i, c in enumerate(CLASS_NAMES)})

    weights = None
    if a.weighted_loss:
        counts = np.bincount(tr.y, minlength=len(CLASS_NAMES)).clip(min=1)
        w = (len(tr) / (len(CLASS_NAMES) * counts)) ** 0.5
        weights = torch.tensor(w, dtype=torch.float)
        print("class weights:", dict(zip(CLASS_NAMES, np.round(w, 2))))

    args = TrainingArguments(
        output_dir=str(Path(a.ckpt_dir) / f"ckpt_{name}"),  # local disk, deleted after the run
        num_train_epochs=a.epochs,
        learning_rate=a.lr,
        per_device_train_batch_size=a.batch,
        per_device_eval_batch_size=a.batch * 2,
        gradient_accumulation_steps=a.grad_accum,
        weight_decay=0.01,
        warmup_steps=int(0.1 * math.ceil(len(ds_tr) / (a.batch * a.grad_accum)) * a.epochs),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=a.seed,
        logging_steps=20,
    )
    trainer = WeightedTrainer(
        class_weights=weights, model=model, args=args,
        train_dataset=ds_tr, eval_dataset=ds_va,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
    trainer.train()

    result = {"name": name, "model": a.model, "max_len": a.max_len, "lr": a.lr,
              "epochs_requested": a.epochs, "batch": a.batch,
              "grad_accum": a.grad_accum, "weighted_loss": a.weighted_loss,
              "seed": a.seed, "version": a.version}
    labels = list(range(len(CLASS_NAMES)))
    for split, ds, df in (("val", ds_va, va), ("test", ds_te, te)):
        logits = trainer.predict(ds).predictions
        pred = logits.argmax(-1)
        result[split] = {
            "accuracy": round(accuracy_score(df.y, pred), 4),
            "macro_f1": round(f1_score(df.y, pred, average="macro", zero_division=0), 4),
            "weighted_f1": round(f1_score(df.y, pred, average="weighted", zero_division=0), 4)}
        if split == "test":
            result["per_class_test"] = classification_report(
                df.y, pred, labels=labels, target_names=CLASS_NAMES,
                zero_division=0, output_dict=True)
            result["confusion_test"] = confusion_matrix(df.y, pred, labels=labels).tolist()
            err = df.assign(pred=[CLASS_NAMES[p] for p in pred])
            err["text"] = err.text.str.slice(0, 300)
            err[["label_name", "pred", "text"]].to_csv(out / f"preds_{name}.csv", index=False)

    (out / f"result_{name}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("name", "val", "test")}, indent=2))
    if a.save_dir:
        trainer.save_model(a.save_dir)
        tok.save_pretrained(a.save_dir)
    shutil.rmtree(args.output_dir, ignore_errors=True)  # checkpoints are ~1 GB each


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Davlan/afro-xlmr-base")
    p.add_argument("--version", choices=["raw", "cleaned"], default="raw")
    p.add_argument("--data_dir", default="data/processed")
    p.add_argument("--out_dir", default="results/transformers")
    p.add_argument("--name", default=None)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--weighted_loss", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", default=None, help="save final model here")
    p.add_argument("--ckpt_dir", default="/tmp/ckpts", help="temporary checkpoint folder (local disk)")
    p.add_argument("--smoke", action="store_true", help="tiny fast run to test the setup")
    main(p.parse_args())
