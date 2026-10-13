"""
Train local models script.
Trains Baseline and BiLSTM models locally on the enriched arXiv dataset,
and compiles comparative evaluation reports.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

from src.data_loader import load_dataset
from src.dataset_preparation import split_dataset
from src.train_baseline import train_baseline_model
from src.train_bilstm import train_bilstm_model
from src.evaluate_models import compile_model_comparison
from config import BASELINE_MODEL_DIR, BILSTM_MODEL_DIR, SCIBERT_MODEL_DIR

print("Step 1: Loading dataset...")
df, metadata = load_dataset()
print(f"Dataset status: {metadata['status']}, Records: {len(df)}")
print(f"Class distribution: {metadata.get('class_distribution')}")

print("\nStep 2: Splitting dataset into stratified train/val/test splits...")
train_df, val_df, test_df, split_meta = split_dataset(df, test_size=0.15, val_size=0.15)
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

print("\nStep 3: Training Baseline (TF-IDF + Logistic Regression)...")
base_pipe, base_metrics = train_baseline_model(train_df, val_df, test_df)

print("\nStep 4: Training BiLSTM...")
bilstm_model, bilstm_tok, bilstm_metrics = train_bilstm_model(
    train_df, val_df, test_df, epochs=15, batch_size=16
)

print("\nStep 5: Compiling Model Comparisons...")
comparison = compile_model_comparison()
print("\nModel Comparison Summary:")
for row in comparison.get("summary_table", []):
    print(f" - {row['Model']}: Macro F1 = {row['Macro F1']:.4f}, Accuracy = {row['Accuracy']:.4f}, Latency = {row['Inference Latency (ms/paper)']}ms")

print(f"\nBest model: {comparison.get('best_model', {}).get('name')}")
print("Local model training pipeline complete!")
