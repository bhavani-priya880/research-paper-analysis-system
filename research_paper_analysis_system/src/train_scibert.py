"""
SciBERT Fine-Tuning and Inference module.
Implements fine-tuning of scientific-text Transformer (allenai/scibert_scivocab_uncased)
for multi-class research paper domain classification.
Includes class-weighted CrossEntropyLoss, sliding-window chunk aggregation for long papers,
confidence thresholding, and documented trade-offs.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import SCIBERT_MODEL_DIR, SCIBERT_CONFIG, RANDOM_SEED, UNCERTAINTY_THRESHOLD
from src.text_preprocessing import prepare_for_scibert
from src.dataset_preparation import prepare_label_mappings, save_label_mapping, compute_balanced_class_weights

# Transformers imports
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, get_linear_schedule_with_warmup


class SciBERTDataset(Dataset):
    """PyTorch Dataset for SciBERT tokenized sequences."""
    def __init__(self, encodings: Dict[str, torch.Tensor], labels: List[int]):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def aggregate_chunk_predictions(
    model: nn.Module,
    tokenizer,
    full_text: str,
    device: torch.device,
    chunk_size: int = 512,
    chunk_overlap: int = 64
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Long Document Strategy: Sliding-Window Chunk Aggregation.
    
    Splits long academic papers into overlapping token chunks of length `chunk_size`
    with `chunk_overlap`. Computes logits for each chunk and computes mean logit pooling.
    
    Mathematical formulation:
        \\bar{z} = \\frac{1}{M} \\sum_{m=1}^M z_m
        p_c = \\frac{\\exp(\\bar{z}_c)}{\\sum_k \\exp(\\bar{z}_k)}
        
    Documented Limitations:
    - Uniform weighting: Treats methodology and reference chunks with equal weight.
    - Computational complexity: Latency scales linearly O(M) with document page count.
    """
    model.eval()
    cleaned = prepare_for_scibert(full_text)

    # Tokenize without truncation to obtain all token ids
    all_tokens = tokenizer.encode(cleaned, add_special_tokens=False)
    stride = max(1, chunk_size - chunk_overlap - 2)  # -2 accounts for [CLS] and [SEP]

    if len(all_tokens) <= (chunk_size - 2):
        # Document fits within a single sequence window
        inputs = tokenizer(
            cleaned,
            max_length=chunk_size,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.cpu().numpy()[0]
            probs = torch.softmax(torch.tensor(logits), dim=0).numpy()
        return probs, {
            "num_chunks": 1,
            "strategy": "Single-window classification",
            "total_tokens": len(all_tokens)
        }

    # Generate overlapping chunks
    chunk_input_ids = []
    chunk_attention_masks = []
    cls_id = tokenizer.cls_token_id or 101
    sep_id = tokenizer.sep_token_id or 102
    pad_id = tokenizer.pad_token_id or 0

    for start_idx in range(0, len(all_tokens), stride):
        sub_tokens = all_tokens[start_idx : start_idx + (chunk_size - 2)]
        if not sub_tokens:
            break
        # Form sequence: [CLS] + sub_tokens + [SEP]
        seq = [cls_id] + sub_tokens + [sep_id]
        attn = [1] * len(seq)
        if len(seq) < chunk_size:
            pad_len = chunk_size - len(seq)
            seq += [pad_id] * pad_len
            attn += [0] * pad_len
        chunk_input_ids.append(seq)
        chunk_attention_masks.append(attn)

        # Cap at max 12 chunks (approx 6,000 words / 15 pages) for computational efficiency
        if len(chunk_input_ids) >= 12:
            break

    input_tensor = torch.tensor(chunk_input_ids, dtype=torch.long).to(device)
    attn_tensor = torch.tensor(chunk_attention_masks, dtype=torch.long).to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_tensor, attention_mask=attn_tensor)
        chunk_logits = outputs.logits.cpu().numpy()  # shape (M, num_classes)

    # Logit mean pooling across chunks
    mean_logits = np.mean(chunk_logits, axis=0)
    # Calibrated softmax
    exp_logits = np.exp(mean_logits - np.max(mean_logits))
    pooled_probs = exp_logits / np.sum(exp_logits)

    return pooled_probs, {
        "num_chunks": len(chunk_input_ids),
        "strategy": "Hierarchical sliding-window logit mean pooling",
        "total_tokens": len(all_tokens)
    }


def train_scibert_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Optional[Path] = None,
    model_name: Optional[str] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None
) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer, Dict[str, Any]]:
    """
    Fine-tunes SciBERT (allenai/scibert_scivocab_uncased) on research paper domain classification.
    Implements class-weighted loss, linear warmup, and evaluation metrics export.
    """
    if output_dir is None:
        output_dir = SCIBERT_MODEL_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"SciBERT Fine-Tuning running on device: {device}")

    # Prepare labels
    all_domains = train_df["domain"].unique().tolist()
    label2id, id2label = prepare_label_mappings(all_domains)
    save_label_mapping(label2id, output_dir / "label_mapping.json")
    num_classes = len(label2id)

    # Initialize Tokenizer and Model
    chosen_model_name = model_name or SCIBERT_CONFIG["model_name"]
    try:
        print(f"Loading SciBERT tokenizer and model from: {chosen_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(chosen_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            chosen_model_name,
            num_labels=num_classes,
            id2label={i: d for d, i in label2id.items()},
            label2id=label2id
        )
    except Exception as e:
        fallback = SCIBERT_CONFIG["fallback_model_name"]
        print(f"Failed to load {chosen_model_name} ({e}). Falling back to: {fallback}")
        chosen_model_name = fallback
        tokenizer = AutoTokenizer.from_pretrained(chosen_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            chosen_model_name,
            num_labels=num_classes,
            id2label={i: d for d, i in label2id.items()},
            label2id=label2id
        )

    model.to(device)

    # Format texts for transformer (title + summary provides concentrated domain signal)
    def prep_texts(df):
        return [
            prepare_for_scibert(f"{t}. {s}")
            for t, s in zip(df["title"], df["summary"])
        ]

    train_texts = prep_texts(train_df)
    val_texts = prep_texts(val_df)
    test_texts = prep_texts(test_df)

    max_len = SCIBERT_CONFIG["max_seq_len"]
    train_encodings = tokenizer(train_texts, max_length=max_len, padding="max_length", truncation=True, return_tensors="pt")
    val_encodings = tokenizer(val_texts, max_length=max_len, padding="max_length", truncation=True, return_tensors="pt")
    test_encodings = tokenizer(test_texts, max_length=max_len, padding="max_length", truncation=True, return_tensors="pt")

    train_labels = [label2id[d] for d in train_df["domain"]]
    val_labels = [label2id[d] for d in val_df["domain"]]
    test_labels = [label2id[d] for d in test_df["domain"]]

    bs = batch_size or SCIBERT_CONFIG["batch_size"]
    train_loader = DataLoader(SciBERTDataset(train_encodings, train_labels), batch_size=bs, shuffle=True)
    val_loader = DataLoader(SciBERTDataset(val_encodings, val_labels), batch_size=bs, shuffle=False)
    test_loader = DataLoader(SciBERTDataset(test_encodings, test_labels), batch_size=bs, shuffle=False)

    # Class-weighted loss
    class_weights = compute_balanced_class_weights(train_labels, num_classes)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)

    # Optimizer & Scheduler
    lr = learning_rate or SCIBERT_CONFIG["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=SCIBERT_CONFIG["weight_decay"])
    
    num_epochs = epochs or SCIBERT_CONFIG["epochs"]
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * SCIBERT_CONFIG["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    best_val_f1 = -1.0
    best_model_state = None

    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(batch["labels"].numpy())

        _, _, val_f1, _ = precision_recall_fscore_support(val_targets, val_preds, average="macro", zero_division=0)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {total_loss/len(train_loader):.4f} - Val Macro F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    train_time = round(time.time() - start_time, 2)

    # Restore best checkpoint
    if best_model_state:
        model.load_state_dict(best_model_state)
        model.to(device)

    # Test Evaluation
    model.eval()
    test_preds, test_probs = [], []
    infer_start = time.time()
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            test_preds.extend(preds)
            test_probs.extend(probs)
    infer_time = round((time.time() - infer_start) / max(len(test_labels), 1) * 1000, 2)

    accuracy = float(accuracy_score(test_labels, test_preds))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(test_labels, test_preds, average="macro", zero_division=0)
    p_per_class, r_per_class, f1_per_class, support = precision_recall_fscore_support(test_labels, test_preds, average=None, zero_division=0)

    class_names = [id2label[i] for i in range(num_classes)]
    per_class_metrics = {}
    for idx, cname in enumerate(class_names):
        per_class_metrics[cname] = {
            "precision": round(float(p_per_class[idx]), 4),
            "recall": round(float(r_per_class[idx]), 4),
            "f1_score": round(float(f1_per_class[idx]), 4),
            "support": int(support[idx])
        }

    cm = confusion_matrix(test_labels, test_preds).tolist()

    metrics = {
        "model_name": f"SciBERT ({chosen_model_name})",
        "model_type": "transformer",
        "accuracy": round(accuracy, 4),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": cm,
        "classes": class_names,
        "training_time_seconds": train_time,
        "inference_time_ms_per_paper": infer_time,
        "num_train_samples": len(train_df),
        "num_test_samples": len(test_df)
    }

    # Save artifacts
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics_path = output_dir / "scibert_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"SciBERT training complete! Macro F1: {metrics['macro_f1']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
    return model, tokenizer, metrics
