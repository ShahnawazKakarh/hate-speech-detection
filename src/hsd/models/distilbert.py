"""DistilBERT fine-tuning for binary hate-speech classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


@dataclass
class DistilBertConfig:
    model_name: str = "distilbert-base-uncased"
    max_length: int = 128
    learning_rate: float = 2e-5
    batch_size: int = 32
    eval_batch_size: int = 64
    epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42
    fp16: bool = False  # set True on CUDA
    output_dir: str = "artifacts/distilbert"
    report_to: str = "none"  # "wandb" to log


class _TextDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int):
        self.enc = tokenizer(
            texts,
            truncation=True,
            padding=False,  # dynamic padding in collator
            max_length=max_length,
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.enc["input_ids"][idx],
            "attention_mask": self.enc["attention_mask"][idx],
            "labels": int(self.labels[idx]),
        }


def _metrics(eval_pred):
    from sklearn.metrics import f1_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_binary": f1_score(labels, preds, average="binary", pos_label=1),
    }


def _trainer_kwargs_for_tokenizer(tokenizer) -> dict:
    """Pass the tokenizer under whichever kwarg this version of transformers supports.

    transformers >=4.46 renamed Trainer's ``tokenizer`` arg to ``processing_class``.
    """
    import inspect

    sig = inspect.signature(Trainer.__init__)
    if "processing_class" in sig.parameters:
        return {"processing_class": tokenizer}
    return {"tokenizer": tokenizer}


class DistilBertClassifier:
    def __init__(self, cfg: DistilBertConfig):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model: AutoModelForSequenceClassification | None = None
        self.trainer: Trainer | None = None

    # ------------------------------------------------------------------ #
    def fit(
        self,
        train_texts: list[str],
        train_labels: list[int],
        val_texts: list[str] | None = None,
        val_labels: list[int] | None = None,
    ) -> DistilBertClassifier:
        c = self.cfg
        self.model = AutoModelForSequenceClassification.from_pretrained(c.model_name, num_labels=2)

        train_ds = _TextDataset(train_texts, train_labels, self.tokenizer, c.max_length)
        val_ds = (
            _TextDataset(val_texts, val_labels, self.tokenizer, c.max_length)
            if val_texts is not None
            else None
        )

        from transformers import DataCollatorWithPadding

        args = TrainingArguments(
            output_dir=c.output_dir,
            num_train_epochs=c.epochs,
            learning_rate=c.learning_rate,
            per_device_train_batch_size=c.batch_size,
            per_device_eval_batch_size=c.eval_batch_size,
            weight_decay=c.weight_decay,
            warmup_ratio=c.warmup_ratio,
            eval_strategy="epoch" if val_ds is not None else "no",
            save_strategy="epoch" if val_ds is not None else "no",
            load_best_model_at_end=val_ds is not None,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            logging_steps=50,
            fp16=c.fp16 and torch.cuda.is_available(),
            seed=c.seed,
            report_to=c.report_to,
            save_total_limit=1,
        )

        self.trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=DataCollatorWithPadding(self.tokenizer),
            compute_metrics=_metrics if val_ds is not None else None,
            **_trainer_kwargs_for_tokenizer(self.tokenizer),
        )
        self.trainer.train()
        return self

    # ------------------------------------------------------------------ #
    def _encode(self, texts: list[str]):
        return self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.cfg.max_length,
            return_tensors="pt",
        )

    @torch.no_grad()
    def predict_proba(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        assert self.model is not None
        device = next(self.model.parameters()).device
        self.model.eval()
        probs = []
        for i in range(0, len(texts), batch_size):
            batch = self._encode(texts[i : i + batch_size]).to(device)
            logits = self.model(**batch).logits
            p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            probs.append(p)
        return np.concatenate(probs)

    def predict(self, texts: list[str], threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(texts) >= threshold).astype(int)

    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: str | Path, cfg: DistilBertConfig | None = None) -> DistilBertClassifier:
        path = Path(path)
        cfg = cfg or DistilBertConfig()
        cfg.model_name = str(path)
        obj = cls(cfg)
        obj.model = AutoModelForSequenceClassification.from_pretrained(path)
        return obj
