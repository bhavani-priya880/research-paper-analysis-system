"""
BiLSTM Model Architecture, Training, and Evaluation module.
Implements a Bidirectional LSTM neural network for multi-class research domain classification.
Includes custom vocabulary management, sequence padding/truncation, class weighting,
early stopping, checkpoint saving, and evaluation metrics.
"""

import json
import time
import os
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
from config import BILSTM_MODEL_DIR, BILSTM_CONFIG, RANDOM_SEED
from src.text_preprocessing import BiLSTMTokenizer, clean_text_for_nlp
from src.dataset_preparation import prepare_label_mappings, save_label_mapping, compute_balanced_class_weights


class ResearchPaperBiLSTM(nn.Module):
    """
    Bidirectional LSTM Classifier with trainable embedding layer,
    multi-layer bidirectional LSTM, dropout regularization, and classification head.
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_classes: int = 6,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0
    ):
        super(ResearchPaperBiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc_dropout = nn.Dropout(dropout)
        # Bidirectional concatenates forward and backward hidden states
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        lstm_out, (hn, cn) = self.lstm(embedded)
        # hn shape: (num_layers * 2, batch_size, hidden_dim)
        # Concatenate forward and backward final hidden states
        forward_hidden = hn[-2, :, :]
        backward_hidden = hn[-1, :, :]
        cat_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)
        dropped = self.fc_dropout(cat_hidden)
        logits = self.fc(dropped)
        return logits


class BiLSTMDataset(Dataset):
    """PyTorch Dataset for token-indexed sequences and target class labels."""
    def __init__(self, sequences: List[List[int]], labels: List[int]):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


def train_bilstm_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Optional[Path] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None
) -> Tuple[ResearchPaperBiLSTM, BiLSTMTokenizer, Dict[str, Any]]:
    """
    Trains and evaluates the BiLSTM model on research paper domains.
    """
    if output_dir is None:
        output_dir = BILSTM_MODEL_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"BiLSTM Training executing on device: {device}")

    # Prepare labels
    all_domains = train_df["domain"].unique().tolist()
    label2id, id2label = prepare_label_mappings(all_domains)
    save_label_mapping(label2id, output_dir / "label_mapping.json")
    num_classes = len(label2id)

    # Build Vocabulary
    train_texts = [f"{t}. {s}" for t, s in zip(train_df["title"], train_df["summary"])]
    val_texts = [f"{t}. {s}" for t, s in zip(val_df["title"], val_df["summary"])]
    test_texts = [f"{t}. {s}" for t, s in zip(test_df["title"], test_df["summary"])]

    tokenizer = BiLSTMTokenizer(
        max_vocab_size=BILSTM_CONFIG["vocab_size"],
        min_freq=2
    )
    tokenizer.build_vocab(train_texts)

    # Save tokenizer vocabulary
    with open(output_dir / "tokenizer_vocab.json", "w", encoding="utf-8") as f:
        json.dump(tokenizer.to_dict(), f, indent=2)

    max_seq_len = BILSTM_CONFIG["max_seq_len"]
    train_seqs = [tokenizer.encode(t, max_seq_len=max_seq_len) for t in train_texts]
    val_seqs = [tokenizer.encode(t, max_seq_len=max_seq_len) for t in val_texts]
    test_seqs = [tokenizer.encode(t, max_seq_len=max_seq_len) for t in test_texts]

    train_labels = [label2id[d] for d in train_df["domain"]]
    val_labels = [label2id[d] for d in val_df["domain"]]
    test_labels = [label2id[d] for d in test_df["domain"]]

    bs = batch_size or BILSTM_CONFIG["batch_size"]
    train_loader = DataLoader(BiLSTMDataset(train_seqs, train_labels), batch_size=bs, shuffle=True)
    val_loader = DataLoader(BiLSTMDataset(val_seqs, val_labels), batch_size=bs, shuffle=False)
    test_loader = DataLoader(BiLSTMDataset(test_seqs, test_labels), batch_size=bs, shuffle=False)

    # Calculate class weights for loss function
    class_weights = compute_balanced_class_weights(train_labels, num_classes)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    # Initialize model
    model = ResearchPaperBiLSTM(
        vocab_size=len(tokenizer.word2idx),
        embedding_dim=BILSTM_CONFIG["embedding_dim"],
        hidden_dim=BILSTM_CONFIG["hidden_dim"],
        num_classes=num_classes,
        num_layers=BILSTM_CONFIG["num_layers"],
        dropout=BILSTM_CONFIG["dropout"],
        pad_idx=tokenizer.pad_idx
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=BILSTM_CONFIG["learning_rate"])

    num_epochs = epochs or BILSTM_CONFIG["epochs"]
    best_val_f1 = -1.0
    best_val_loss = float("inf")
    patience = BILSTM_CONFIG["patience"]
    patience_counter = 0
    best_model_state = None

    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for seqs_batch, labels_batch in train_loader:
            seqs_batch, labels_batch = seqs_batch.to(device), labels_batch.to(device)
            optimizer.zero_grad()
            logits = model(seqs_batch)
            loss = criterion(logits, labels_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation phase
        model.eval()
        val_preds, val_targets = [], []
        val_loss = 0.0
        with torch.no_grad():
            for seqs_batch, labels_batch in val_loader:
                seqs_batch, labels_batch = seqs_batch.to(device), labels_batch.to(device)
                logits = model(seqs_batch)
                loss = criterion(logits, labels_batch)
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(labels_batch.cpu().numpy())

        avg_val_loss = val_loss / max(len(val_loader), 1)
        _, _, val_f1, _ = precision_recall_fscore_support(val_targets, val_preds, average="macro", zero_division=0)
        val_acc = accuracy_score(val_targets, val_preds)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {total_loss/len(train_loader):.4f} - Val Loss: {avg_val_loss:.4f} - Val Acc: {val_acc:.4f} - Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1 or (val_f1 == best_val_f1 and avg_val_loss < best_val_loss):
            best_val_f1 = max(val_f1, best_val_f1)
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    train_time = round(time.time() - start_time, 2)

    # Load best checkpoint
    if best_model_state:
        model.load_state_dict(best_model_state)

    # Test evaluation
    model.eval()
    test_preds, test_probs = [], []
    infer_start = time.time()
    with torch.no_grad():
        for seqs_batch, _ in test_loader:
            seqs_batch = seqs_batch.to(device)
            logits = model(seqs_batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
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
        "model_name": "BiLSTM (Bidirectional LSTM)",
        "model_type": "deep_learning",
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
    model_path = output_dir / "bilstm_model.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "vocab_size": len(tokenizer.word2idx),
        "embedding_dim": BILSTM_CONFIG["embedding_dim"],
        "hidden_dim": BILSTM_CONFIG["hidden_dim"],
        "num_classes": num_classes,
        "num_layers": BILSTM_CONFIG["num_layers"],
        "dropout": BILSTM_CONFIG["dropout"],
        "pad_idx": tokenizer.pad_idx
    }, model_path)

    metrics_path = output_dir / "bilstm_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"BiLSTM training complete! Macro F1: {metrics['macro_f1']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
    return model, tokenizer, metrics
