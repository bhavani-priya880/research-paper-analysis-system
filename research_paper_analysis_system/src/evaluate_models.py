"""
Model Evaluation and Comparison module.
Aggregates metrics across Baseline, BiLSTM, and SciBERT models on identical held-out test sets.
Computes Macro F1, Accuracy, Precision, Recall, Per-class scores, confusion matrices, and latency.
Identifies the best-performing model based on class-balanced Macro F1.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import BASELINE_MODEL_DIR, BILSTM_MODEL_DIR, SCIBERT_MODEL_DIR, REPORTS_DIR


def load_model_metrics(model_dir: Path, metric_filename: str) -> Optional[Dict[str, Any]]:
    """Loads metrics JSON file if available."""
    metrics_path = model_dir / metric_filename
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def compile_model_comparison(
    baseline_metrics: Optional[Dict[str, Any]] = None,
    bilstm_metrics: Optional[Dict[str, Any]] = None,
    scibert_metrics: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Consolidates evaluation metrics across Baseline, BiLSTM, and SciBERT.
    Identifies the best-performing model according to Macro F1-score.
    """
    # Load from disk if not explicitly passed
    if baseline_metrics is None:
        baseline_metrics = load_model_metrics(BASELINE_MODEL_DIR, "baseline_metrics.json")
    if bilstm_metrics is None:
        bilstm_metrics = load_model_metrics(BILSTM_MODEL_DIR, "bilstm_metrics.json")
    if scibert_metrics is None:
        scibert_metrics = load_model_metrics(SCIBERT_MODEL_DIR, "scibert_metrics.json")

    models_data = []
    for m in [baseline_metrics, bilstm_metrics, scibert_metrics]:
        if m:
            models_data.append(m)

    if not models_data:
        return {
            "status": "no_metrics",
            "message": "No trained model metrics found. Train baseline, BiLSTM, and SciBERT models first."
        }

    # Comparison summary table
    summary_rows = []
    for m in models_data:
        summary_rows.append({
            "Model": m.get("model_name", "Unknown"),
            "Architecture": m.get("model_type", "Standard"),
            "Accuracy": m.get("accuracy", 0.0),
            "Macro Precision": m.get("macro_precision", 0.0),
            "Macro Recall": m.get("macro_recall", 0.0),
            "Macro F1": m.get("macro_f1", 0.0),
            "Training Time (s)": m.get("training_time_seconds", 0.0),
            "Inference Latency (ms/paper)": m.get("inference_time_ms_per_paper", 0.0)
        })

    summary_df = pd.DataFrame(summary_rows)

    # Determine best model based on Macro F1
    best_row = summary_df.loc[summary_df["Macro F1"].idxmax()]
    best_model_name = best_row["Model"]
    best_f1 = best_row["Macro F1"]

    # Per-class comparison
    all_classes = set()
    for m in models_data:
        all_classes.update(m.get("per_class_metrics", {}).keys())
    sorted_classes = sorted(list(all_classes))

    per_class_comparison = []
    for cname in sorted_classes:
        row = {"Domain": cname}
        for m in models_data:
            mname = m.get("model_name", "Unknown")
            c_metrics = m.get("per_class_metrics", {}).get(cname, {})
            row[f"{mname} F1"] = c_metrics.get("f1_score", 0.0)
            row[f"{mname} Recall"] = c_metrics.get("recall", 0.0)
        per_class_comparison.append(row)

    comparison_report = {
        "status": "success",
        "summary_table": summary_rows,
        "best_model": {
            "name": best_model_name,
            "metric": "Macro F1-Score",
            "score": best_f1,
            "justification": (
                "Selected based on Macro F1-score. Macro F1 assigns equal weighting to each scientific domain "
                "regardless of class frequency, making it the most rigorous metric for imbalanced multi-domain datasets."
            )
        },
        "per_class_comparison": per_class_comparison,
        "models_evaluated": [m.get("model_name") for m in models_data]
    }

    # Save to reports directory
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_file = REPORTS_DIR / "model_comparison_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(comparison_report, f, indent=2)

    return comparison_report
