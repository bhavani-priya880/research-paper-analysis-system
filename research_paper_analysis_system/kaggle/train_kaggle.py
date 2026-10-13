"""
Kaggle Training Script: NLP-Based Multi-Domain Research Paper Analysis System.
Designed to execute inside a Kaggle Notebook / GPU accelerator (e.g. Tesla P100 / T4 x2).

Input Path (Read-Only):
    /kaggle/input/datasets/sumitm004/arxiv-scientific-research-papers-dataset/
Output Path (Writable):
    /kaggle/working/models/
    /kaggle/working/reports/
    /kaggle/working/models.zip (packaged for local download)
"""

import os
import sys
import shutil
import zipfile
from pathlib import Path
import torch

# Ensure local source directory is importable
KAGGLE_WORKING = Path("/kaggle/working")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data_loader import inspect_dataset_directory, load_dataset
from src.dataset_preparation import split_dataset
from src.train_baseline import train_baseline_model
from src.train_bilstm import train_bilstm_model
from src.train_scibert import train_scibert_model
from src.evaluate_models import compile_model_comparison
from config import KAGGLE_INPUT_DIR, SUPPORTED_DOMAINS


def main():
    print("=" * 70)
    print("NLP-BASED MULTI-DOMAIN RESEARCH PAPER ANALYSIS SYSTEM - KAGGLE TRAINER")
    print("=" * 70)

    # Check CUDA acceleration
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU (No GPU detected)"
    print(f"Compute Hardware: {device_name}")
    print(f"CUDA Available: {cuda_available}")

    # Step 1: Inspect Kaggle input directory
    print("\n--- Step 1: Inspecting Dataset Directory ---")
    inspection = inspect_dataset_directory(KAGGLE_INPUT_DIR)
    print(f"Directory exists: {inspection['exists']}")
    print(f"Total files found: {inspection['total_files']}")
    for f in inspection.get("supported_data_files", []):
        print(f" - Found data file: {f['relative_path']} ({f['size_mb']} MB)")

    # Step 2: Load and Sanitize Dataset
    print("\n--- Step 2: Loading & Sanitizing Dataset ---")
    df, meta = load_dataset(target_domains=SUPPORTED_DOMAINS, max_samples=10000)
    print(f"Dataset status: {meta['status']}")
    print(f"Raw records: {meta.get('total_raw_records')}, Retained records: {meta.get('total_retained')}")
    print(f"Class distribution: {meta.get('class_distribution')}")
    print(f"Class imbalance ratio: {meta.get('class_imbalance_ratio')}")

    if len(df) == 0:
        print("ERROR: No valid records loaded. Please verify dataset mount in Kaggle.")
        return

    # Step 3: Stratified Split with Leakage Prevention
    print("\n--- Step 3: Stratified Train / Val / Test Split ---")
    train_df, val_df, test_df, split_meta = split_dataset(df, test_size=0.15, val_size=0.15)
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")

    # Set writable output directories in /kaggle/working
    out_models_dir = KAGGLE_WORKING / "models"
    out_reports_dir = KAGGLE_WORKING / "reports"
    out_models_dir.mkdir(parents=True, exist_ok=True)
    out_reports_dir.mkdir(parents=True, exist_ok=True)

    base_out = out_models_dir / "baseline"
    bilstm_out = out_models_dir / "bilstm"
    scibert_out = out_models_dir / "scibert"

    # Step 4: Train Baseline Model (TF-IDF + Logistic Regression)
    print("\n--- Step 4: Training Baseline Model (TF-IDF + LogReg) ---")
    base_pipe, base_metrics = train_baseline_model(train_df, val_df, test_df, output_dir=base_out)
    print(f"Baseline Macro F1: {base_metrics['macro_f1']:.4f}, Accuracy: {base_metrics['accuracy']:.4f}")

    # Step 5: Train Comparison Model (BiLSTM)
    print("\n--- Step 5: Training Comparison Model (BiLSTM) ---")
    bilstm_model, bilstm_tok, bilstm_metrics = train_bilstm_model(
        train_df, val_df, test_df,
        output_dir=bilstm_out,
        epochs=12,
        batch_size=32
    )
    print(f"BiLSTM Macro F1: {bilstm_metrics['macro_f1']:.4f}, Accuracy: {bilstm_metrics['accuracy']:.4f}")

    # Step 6: Fine-Tune Proposed Model (SciBERT)
    print("\n--- Step 6: Fine-Tuning Proposed Model (SciBERT) ---")
    scibert_model, scibert_tok, scibert_metrics = train_scibert_model(
        train_df, val_df, test_df,
        output_dir=scibert_out,
        epochs=3,
        batch_size=16 if cuda_available else 4
    )
    print(f"SciBERT Macro F1: {scibert_metrics['macro_f1']:.4f}, Accuracy: {scibert_metrics['accuracy']:.4f}")

    # Step 7: Consolidate Model Evaluation & Comparison
    print("\n--- Step 7: Consolidating Multi-Model Evaluation ---")
    comparison = compile_model_comparison(
        baseline_metrics=base_metrics,
        bilstm_metrics=bilstm_metrics,
        scibert_metrics=scibert_metrics
    )

    print("\n" + "=" * 70)
    print("FINAL BENCHMARK COMPARISON RESULTS")
    print("=" * 70)
    for r in comparison.get("summary_table", []):
        print(f"Model: {r['Model']:<35} | Acc: {r['Accuracy']:.4f} | Macro F1: {r['Macro F1']:.4f} | Latency: {r['Inference Latency (ms/paper)']}ms")

    print(f"\nIdentified Best Model: {comparison['best_model']['name']}")
    print(f"Justification: {comparison['best_model']['justification']}")

    # Step 8: Package Artifacts into models.zip for Local Download
    print("\n--- Step 8: Packaging Artifacts for Local Export ---")
    zip_path = KAGGLE_WORKING / "trained_models_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(out_models_dir):
            for file in files:
                abs_f = Path(root) / file
                rel_f = abs_f.relative_to(KAGGLE_WORKING)
                zf.write(abs_f, arcname=str(rel_f))
        for root, _, files in os.walk(out_reports_dir):
            for file in files:
                abs_f = Path(root) / file
                rel_f = abs_f.relative_to(KAGGLE_WORKING)
                zf.write(abs_f, arcname=str(rel_f))

    print(f"Artifacts successfully packaged: {zip_path} ({round(zip_path.stat().st_size / (1024*1024), 2)} MB)")
    print("\nYou can now download 'trained_models_bundle.zip' from the Kaggle Output tab,")
    print("extract it into your local project root, and run `streamlit run app.py`!")
    print("=" * 70)


if __name__ == "__main__":
    main()
