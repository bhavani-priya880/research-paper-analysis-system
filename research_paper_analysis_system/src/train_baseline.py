"""
Baseline Model Training module.
Implements TF-IDF + Logistic Regression (and Linear SVM) as a classical NLP benchmark.
Evaluates on test split, logs metrics, and exports serializable model pipeline.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import BASELINE_MODEL_DIR, BASELINE_CONFIG, RANDOM_SEED
from src.text_preprocessing import clean_text_for_nlp
from src.dataset_preparation import prepare_label_mappings, save_label_mapping


def train_baseline_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Optional[Path] = None
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Trains TF-IDF + Logistic Regression baseline with balanced class weights.
    Evaluates on test set and saves artifacts.
    """
    if output_dir is None:
        output_dir = BASELINE_MODEL_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare labels
    all_domains = train_df["domain"].unique().tolist()
    label2id, id2label = prepare_label_mappings(all_domains)
    save_label_mapping(label2id, output_dir / "label_mapping.json")

    # Combine title and summary for richer feature representation
    def prep_text(df):
        return [
            clean_text_for_nlp(f"{t}. {s}", remove_stopwords=True)
            for t, s in zip(df["title"], df["summary"])
        ]

    X_train = prep_text(train_df)
    y_train = [label2id[d] for d in train_df["domain"]]

    X_test = prep_text(test_df)
    y_test = [label2id[d] for d in test_df["domain"]]

    # Construct pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=BASELINE_CONFIG["max_features"],
            ngram_range=BASELINE_CONFIG["ngram_range"],
            min_df=BASELINE_CONFIG["min_df"],
            max_df=BASELINE_CONFIG["max_df"],
            sublinear_tf=True
        )),
        ("classifier", LogisticRegression(
            C=BASELINE_CONFIG["c_param"],
            class_weight=BASELINE_CONFIG["class_weight"],
            max_iter=1000,
            random_state=RANDOM_SEED
        ))
    ])

    print("Training TF-IDF + Logistic Regression baseline...")
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    train_time = round(time.time() - start_time, 2)

    # Test evaluation
    infer_start = time.time()
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)
    infer_time = round((time.time() - infer_start) / max(len(X_test), 1) * 1000, 2)  # ms per paper

    accuracy = float(accuracy_score(y_test, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    p_per_class, r_per_class, f1_per_class, support = precision_recall_fscore_support(y_test, y_pred, average=None, zero_division=0)

    class_names = [id2label[i] for i in range(len(label2id))]
    per_class_metrics = {}
    for idx, cname in enumerate(class_names):
        per_class_metrics[cname] = {
            "precision": round(float(p_per_class[idx]), 4),
            "recall": round(float(r_per_class[idx]), 4),
            "f1_score": round(float(f1_per_class[idx]), 4),
            "support": int(support[idx])
        }

    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "model_name": "TF-IDF + Logistic Regression",
        "model_type": "baseline",
        "accuracy": round(accuracy, 4),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": cm,
        "classes": class_names,
        "training_time_seconds": train_time,
        "inference_time_ms_per_paper": infer_time,
        "num_train_samples": len(X_train),
        "num_test_samples": len(X_test)
    }

    # Save artifacts
    pipeline_path = output_dir / "baseline_pipeline.joblib"
    joblib.dump(pipeline, pipeline_path)

    metrics_path = output_dir / "baseline_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Baseline training complete! Macro F1: {metrics['macro_f1']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
    return pipeline, metrics
