"""Unified dataset loader returning train/val/test splits as pandas DataFrames."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from hsd.data.preprocess import clean_series
from hsd.utils import PROCESSED_DIR, get_logger

log = get_logger(__name__)


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    name: str

    def texts(self, split: str) -> list[str]:
        return getattr(self, split)["text"].tolist()

    def labels(self, split: str, three_way: bool = False) -> list[int]:
        col = "label3" if three_way else "label"
        return getattr(self, split)[col].tolist()


def load_dataset(name: str, *, preprocess: bool = True) -> Splits:
    """Load a prepared dataset from ``data/processed/{name}/``.

    Args:
        name: "davidson" or "hatexplain".
        preprocess: apply :func:`hsd.data.preprocess.clean_text` to all texts.
    """
    root = PROCESSED_DIR / name
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset '{name}' not prepared. Run: python -m hsd.data.prepare --dataset {name}"
        )

    def _load(split: str) -> pd.DataFrame:
        df = pd.read_parquet(root / f"{split}.parquet")
        if preprocess:
            df["text"] = clean_series(df["text"])
            df = df[df["text"].str.len() > 0].reset_index(drop=True)
        return df

    splits = Splits(
        train=_load("train"),
        val=_load("val"),
        test=_load("test"),
        name=name,
    )
    log.info(
        "loaded %s: train=%d val=%d test=%d",
        name,
        len(splits.train),
        len(splits.val),
        len(splits.test),
    )
    return splits
