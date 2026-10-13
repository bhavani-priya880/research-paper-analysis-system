"""
Dataset Preparation module.
Handles stratified train/val/test splitting, cross-split leakage prevention,
class weighting for handling imbalanced datasets, and label encoding.
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import RANDOM_SEED, MODELS_DIR


def prepare_label_mappings(domains: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Creates deterministic bi-directional label mappings.
    Sorts domains alphabetically for reproducibility.
    """
    unique_domains = sorted(list(set(domains)))
    label2id = {dom: idx for idx, dom in enumerate(unique_domains)}
    id2label = {idx: dom for idx, dom in enumerate(unique_domains)}
    return label2id, id2label


def save_label_mapping(label2id: Dict[str, int], filepath: Optional[Path] = None):
    """Saves label mapping to JSON."""
    if filepath is None:
        filepath = MODELS_DIR / "label_mapping.json"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(label2id, f, indent=2)


def load_label_mapping(filepath: Optional[Path] = None) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Loads label mapping from JSON."""
    if filepath is None:
        filepath = MODELS_DIR / "label_mapping.json"
    if not filepath.exists():
        raise FileNotFoundError(f"Label mapping file not found at: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        label2id = json.load(f)
    id2label = {int(v): k for k, v in label2id.items()}
    return label2id, id2label


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_seed: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Splits dataset into stratified train, validation, and test splits
    with verified cross-split leakage prevention.
    
    Returns:
        (train_df, val_df, test_df, split_metadata)
    """
    if "domain" not in df.columns or "title" not in df.columns:
        raise ValueError("DataFrame must contain 'domain' and 'title' columns.")

    # Deduplicate titles to prevent duplicate papers from appearing in multiple splits
    initial_count = len(df)
    dedup_df = df.drop_duplicates(subset=["title"]).reset_index(drop=True)
    dropped_dupes = initial_count - len(dedup_df)

    # Filter out classes with too few instances for stratified splitting (< 3 samples)
    class_counts = dedup_df["domain"].value_counts()
    valid_classes = class_counts[class_counts >= 3].index
    filtered_df = dedup_df[dedup_df["domain"].isin(valid_classes)].copy().reset_index(drop=True)

    y = filtered_df["domain"].values

    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        filtered_df,
        test_size=test_size,
        stratify=y,
        random_state=random_seed
    )

    # Second split: train vs val
    val_ratio_adjusted = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_adjusted,
        stratify=train_val_df["domain"].values,
        random_state=random_seed
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Verify no title overlap (leakage safety assertion)
    train_titles = set(train_df["title"].str.lower())
    val_titles = set(val_df["title"].str.lower())
    test_titles = set(test_df["title"].str.lower())

    assert len(train_titles.intersection(val_titles)) == 0, "Data leakage detected between train and val!"
    assert len(train_titles.intersection(test_titles)) == 0, "Data leakage detected between train and test!"
    assert len(val_titles.intersection(test_titles)) == 0, "Data leakage detected between val and test!"

    split_metadata = {
        "total_samples": len(filtered_df),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "dropped_duplicates": dropped_dupes,
        "classes": sorted(list(valid_classes)),
        "train_distribution": train_df["domain"].value_counts().to_dict(),
        "val_distribution": val_df["domain"].value_counts().to_dict(),
        "test_distribution": test_df["domain"].value_counts().to_dict()
    }

    return train_df, val_df, test_df, split_metadata


def compute_balanced_class_weights(
    train_labels: List[int],
    num_classes: int
) -> np.ndarray:
    """
    Computes balanced class weights using sklearn's balanced weighting formula:
    n_samples / (n_classes * np.bincount(y))
    """
    classes = np.arange(num_classes)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_labels
    )
    return weights.astype(np.float32)
