"""Download + prepare Davidson and HateXplain datasets.

Outputs unified parquet files at ``data/processed/{dataset}/{split}.parquet``
with columns ``text``, ``label`` (0=non-hate, 1=hate), ``label3`` (0=hate,
1=offensive, 2=neither/normal).

Usage:
    python -m hsd.data.prepare --dataset davidson
    python -m hsd.data.prepare --dataset hatexplain
    python -m hsd.data.prepare --dataset all
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from hsd.utils import PROCESSED_DIR, RAW_DIR, ensure_dir, get_logger, set_seed

log = get_logger(__name__)

DAVIDSON_URL = (
    "https://raw.githubusercontent.com/t-davidson/"
    "hate-speech-and-offensive-language/master/data/labeled_data.csv"
)
HATEXPLAIN_URL = "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/dataset.json"
HATEXPLAIN_SPLIT_URL = (
    "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/post_id_divisions.json"
)


def _download(url: str, dest: Path) -> Path:
    ensure_dir(dest.parent)
    if dest.exists():
        log.info("cached: %s", dest)
        return dest
    log.info("downloading %s", url)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


# --------------------------------------------------------------------------- #
# Davidson
# --------------------------------------------------------------------------- #
def prepare_davidson(seed: int = 42) -> None:
    """Davidson labels: 0=hate, 1=offensive, 2=neither.

    Binary label: hate (orig 0) -> 1, else 0.
    """
    raw = _download(DAVIDSON_URL, RAW_DIR / "davidson" / "labeled_data.csv")
    df = pd.read_csv(raw)
    df = df.rename(columns={"class": "label3", "tweet": "text"})
    df["label"] = (df["label3"] == 0).astype(int)
    df = df[["text", "label", "label3"]].dropna().reset_index(drop=True)

    train, tmp = train_test_split(df, test_size=0.2, random_state=seed, stratify=df["label3"])
    val, test = train_test_split(tmp, test_size=0.5, random_state=seed, stratify=tmp["label3"])

    out = ensure_dir(PROCESSED_DIR / "davidson")
    train.to_parquet(out / "train.parquet", index=False)
    val.to_parquet(out / "val.parquet", index=False)
    test.to_parquet(out / "test.parquet", index=False)
    log.info(
        "davidson prepared: train=%d val=%d test=%d  hate_pct=%.2f%%",
        len(train),
        len(val),
        len(test),
        100 * df["label"].mean(),
    )


# --------------------------------------------------------------------------- #
# HateXplain
# --------------------------------------------------------------------------- #
def _hatexplain_majority(annotators: list[dict]) -> str:
    """Majority vote over the 3 annotators' labels."""
    votes = [a["label"] for a in annotators]
    return max(set(votes), key=votes.count)


def prepare_hatexplain() -> None:
    """HateXplain labels: hatespeech / offensive / normal.

    Maps to label3 (0=hate, 1=offensive, 2=normal) for consistency with Davidson,
    and binary label (hate=1, else=0). Uses the official train/val/test split.
    """
    raw = _download(HATEXPLAIN_URL, RAW_DIR / "hatexplain" / "dataset.json")
    splits = _download(HATEXPLAIN_SPLIT_URL, RAW_DIR / "hatexplain" / "post_id_divisions.json")

    data = json.loads(raw.read_text())
    split_map = json.loads(splits.read_text())

    label_to_3 = {"hatespeech": 0, "offensive": 1, "normal": 2}

    rows = []
    for pid, post in data.items():
        majority = _hatexplain_majority(post["annotators"])
        if majority not in label_to_3:
            continue
        text = " ".join(post["post_tokens"])
        rows.append(
            {
                "post_id": pid,
                "text": text,
                "label3": label_to_3[majority],
                "label": int(majority == "hatespeech"),
            }
        )
    df = pd.DataFrame(rows)

    out = ensure_dir(PROCESSED_DIR / "hatexplain")
    for split_name in ("train", "val", "test"):
        key = split_name if split_name in split_map else split_name.replace("val", "validation")
        ids = set(split_map.get(key, []))
        sub = df[df["post_id"].isin(ids)].drop(columns=["post_id"]).reset_index(drop=True)
        sub.to_parquet(out / f"{split_name}.parquet", index=False)
        log.info("hatexplain %s: %d rows", split_name, len(sub))


@click.command()
@click.option(
    "--dataset",
    type=click.Choice(["davidson", "hatexplain", "all"]),
    required=True,
)
@click.option("--seed", type=int, default=42)
def main(dataset: str, seed: int) -> None:
    set_seed(seed)
    if dataset in ("davidson", "all"):
        prepare_davidson(seed=seed)
    if dataset in ("hatexplain", "all"):
        prepare_hatexplain()


if __name__ == "__main__":
    main()
