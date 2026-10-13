"""
Unified Domain Prediction module.
Orchestrates inference across SciBERT, BiLSTM, and Baseline models.
Handles long paper chunking, confidence score calibration, domain restriction/filtering modes,
and batch summary aggregation.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import joblib
import torch

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    BASELINE_MODEL_DIR,
    BILSTM_MODEL_DIR,
    SCIBERT_MODEL_DIR,
    UNCERTAINTY_THRESHOLD,
    SUPPORTED_DOMAINS
)
from src.text_preprocessing import clean_text_for_nlp, prepare_for_scibert, BiLSTMTokenizer
from src.train_bilstm import ResearchPaperBiLSTM
from src.train_scibert import aggregate_chunk_predictions

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class PaperDomainPredictor:
    """
    Unified predictor supporting SciBERT, BiLSTM, and TF-IDF Baseline.
    Manages model loading, device routing, thresholding, and domain filtering.
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.baseline_pipeline = None
        self.bilstm_model = None
        self.bilstm_tokenizer = None
        self.scibert_model = None
        self.scibert_tokenizer = None
        self.label2id = None
        self.id2label = None
        self._load_available_models()

    def _load_available_models(self):
        """Loads whatever models have been trained and exported."""
        # 1. Baseline
        baseline_path = BASELINE_MODEL_DIR / "baseline_pipeline.joblib"
        if baseline_path.exists():
            try:
                self.baseline_pipeline = joblib.load(baseline_path)
            except Exception as e:
                print(f"Notice: Baseline model not loaded: {e}")

        # 2. BiLSTM
        bilstm_model_path = BILSTM_MODEL_DIR / "bilstm_model.pt"
        bilstm_vocab_path = BILSTM_MODEL_DIR / "tokenizer_vocab.json"
        if bilstm_model_path.exists() and bilstm_vocab_path.exists():
            try:
                with open(bilstm_vocab_path, "r", encoding="utf-8") as f:
                    vocab_data = json.load(f)
                self.bilstm_tokenizer = BiLSTMTokenizer.from_dict(vocab_data)

                ckpt = torch.load(bilstm_model_path, map_location=self.device)
                self.bilstm_model = ResearchPaperBiLSTM(
                    vocab_size=ckpt["vocab_size"],
                    embedding_dim=ckpt.get("embedding_dim", 128),
                    hidden_dim=ckpt.get("hidden_dim", 128),
                    num_classes=ckpt["num_classes"],
                    num_layers=ckpt.get("num_layers", 2),
                    dropout=ckpt.get("dropout", 0.3),
                    pad_idx=ckpt.get("pad_idx", 0)
                )
                self.bilstm_model.load_state_dict(ckpt["state_dict"])
                self.bilstm_model.to(self.device)
                self.bilstm_model.eval()
            except Exception as e:
                print(f"Notice: BiLSTM model not loaded: {e}")

        # 3. SciBERT
        if TRANSFORMERS_AVAILABLE and SCIBERT_MODEL_DIR.exists():
            try:
                config_file = SCIBERT_MODEL_DIR / "config.json"
                if config_file.exists():
                    self.scibert_tokenizer = AutoTokenizer.from_pretrained(str(SCIBERT_MODEL_DIR))
                    self.scibert_model = AutoModelForSequenceClassification.from_pretrained(str(SCIBERT_MODEL_DIR))
                    self.scibert_model.to(self.device)
                    self.scibert_model.eval()
            except Exception as e:
                print(f"Notice: SciBERT model not loaded: {e}")

        # Load label mapping
        for mdir in [SCIBERT_MODEL_DIR, BILSTM_MODEL_DIR, BASELINE_MODEL_DIR]:
            mapping_file = mdir / "label_mapping.json"
            if mapping_file.exists():
                with open(mapping_file, "r", encoding="utf-8") as f:
                    self.label2id = json.load(f)
                    self.id2label = {int(v): k for k, v in self.label2id.items()}
                break

    def get_available_model_names(self) -> List[str]:
        """Returns list of loaded models ready for inference."""
        models = []
        if self.scibert_model is not None:
            models.append("SciBERT (Proposed)")
        if self.bilstm_model is not None:
            models.append("BiLSTM (Deep Learning)")
        if self.baseline_pipeline is not None:
            models.append("Baseline (TF-IDF + LogReg)")
        return models

    def predict_single_paper(
        self,
        title: str,
        abstract: str,
        full_text: str,
        model_name: str = "SciBERT (Proposed)",
        selected_domains: Optional[List[str]] = None,
        mode: str = "unrestricted"  # "unrestricted" or "restricted"
    ) -> Dict[str, Any]:
        """
        Predicts the research domain for an individual paper.
        
        Args:
            title: Extracted paper title
            abstract: Extracted paper abstract
            full_text: Full document body
            model_name: Chosen model string
            selected_domains: User-selected domain list
            mode: "unrestricted" (predicts among all, filters view) or "restricted" (renormalizes within selected)
        """
        if not self.id2label:
            return {
                "predicted_domain": "Model Not Loaded",
                "confidence": 0.0,
                "alternatives": [],
                "review_status": "Error",
                "strategy_note": "No trained model weights found in models/ directory."
            }

        num_classes = len(self.id2label)
        classes = [self.id2label[i] for i in range(num_classes)]
        probs = np.zeros(num_classes)
        strategy_note = ""

        # Choose text representation:
        # If abstract exists and is informative (> 50 chars), use Title + Abstract.
        # Otherwise, analyze full text.
        has_good_abstract = len(abstract.strip()) >= 50
        summary_text = f"{title}. {abstract}" if has_good_abstract else f"{title}. {full_text[:3000]}"

        # Model dispatch
        if "scibert" in model_name.lower() and self.scibert_model is not None:
            if has_good_abstract or len(full_text.split()) < 400:
                # Classify abstract directly
                cleaned = prepare_for_scibert(summary_text)
                inputs = self.scibert_tokenizer(
                    cleaned,
                    max_length=512,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                with torch.no_grad():
                    outputs = self.scibert_model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
                strategy_note = "Abstract-focused Transformer classification"
            else:
                # Long document chunk aggregation
                probs, chunk_info = aggregate_chunk_predictions(
                    self.scibert_model,
                    self.scibert_tokenizer,
                    full_text,
                    self.device
                )
                strategy_note = f"Sliding-window chunk aggregation ({chunk_info['num_chunks']} chunks)"

        elif "bilstm" in model_name.lower() and self.bilstm_model is not None:
            cleaned = clean_text_for_nlp(summary_text, remove_stopwords=False)
            seq = self.bilstm_tokenizer.encode(cleaned, max_seq_len=256)
            inp = torch.tensor([seq], dtype=torch.long).to(self.device)
            with torch.no_grad():
                logits = self.bilstm_model(inp)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            strategy_note = "BiLSTM bidirectional sequence inference"

        elif self.baseline_pipeline is not None:
            cleaned = clean_text_for_nlp(summary_text, remove_stopwords=True)
            probs = self.baseline_pipeline.predict_proba([cleaned])[0]
            strategy_note = "TF-IDF + Logistic Regression inference"

        else:
            # Fallback if selected model is missing: use whichever is loaded
            if self.baseline_pipeline is not None:
                cleaned = clean_text_for_nlp(summary_text, remove_stopwords=True)
                probs = self.baseline_pipeline.predict_proba([cleaned])[0]
                strategy_note = "Fallback to TF-IDF baseline (selected model not loaded)"
            else:
                return {
                    "predicted_domain": "Untrained",
                    "confidence": 0.0,
                    "alternatives": [],
                    "review_status": "No Model",
                    "strategy_note": "No model available."
                }

        # Handling domain restriction vs unrestricted
        if mode == "restricted" and selected_domains:
            # Mask out non-selected classes and renormalize probabilities
            selected_indices = [i for i, c in enumerate(classes) if c in selected_domains]
            if selected_indices:
                sub_probs = probs[selected_indices]
                sum_p = np.sum(sub_probs)
                if sum_p > 0:
                    sub_probs = sub_probs / sum_p
                # Build modified probability array
                new_probs = np.zeros_like(probs)
                for idx, orig_i in enumerate(selected_indices):
                    new_probs[orig_i] = sub_probs[idx]
                probs = new_probs
                strategy_note += " [Restricted & Renormalized to Selected Domains]"

        # Rank predictions
        ranked_indices = np.argsort(probs)[::-1]
        top_idx = ranked_indices[0]
        top_domain = classes[top_idx]
        top_prob = float(probs[top_idx])

        # Alternatives
        alternatives = []
        for i in ranked_indices[1:4]:
            if probs[i] > 0.01:
                alternatives.append({
                    "domain": classes[i],
                    "confidence": round(float(probs[i]), 4)
                })

        # Uncertainty review status
        if top_prob >= UNCERTAINTY_THRESHOLD:
            review_status = "Confident"
        elif top_prob >= 0.40:
            review_status = "Moderate Confidence"
        else:
            review_status = "Needs Review / Low Confidence"

        return {
            "predicted_domain": top_domain,
            "confidence": round(top_prob, 4),
            "alternatives": alternatives,
            "all_probabilities": {classes[i]: round(float(probs[i]), 4) for i in range(len(classes))},
            "review_status": review_status,
            "strategy_note": strategy_note
        }

    def predict_batch(
        self,
        extracted_papers: List[Dict[str, Any]],
        model_name: str = "SciBERT (Proposed)",
        selected_domains: Optional[List[str]] = None,
        mode: str = "unrestricted"
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Executes prediction across an entire batch of extracted PDFs.
        Computes document-level results and aggregate batch statistics.
        """
        results = []
        domain_counts = {}
        uncertain_papers = []

        for paper in extracted_papers:
            filename = paper.get("filename", "unknown.pdf")
            title = paper.get("title", filename)
            abstract = paper.get("abstract", "")
            full_text = paper.get("full_text", "")
            is_scanned = paper.get("is_scanned", False)
            success = paper.get("success", True)

            if not success or is_scanned or len(full_text.strip()) == 0:
                doc_res = {
                    "filename": filename,
                    "title": title,
                    "predicted_domain": "Unclassifiable (Scanned / Empty)",
                    "confidence": 0.0,
                    "alternatives": [],
                    "review_status": "Scanned / Insufficient Text",
                    "strategy_note": "No extractable text layer available."
                }
                uncertain_papers.append(filename)
            else:
                pred = self.predict_single_paper(
                    title=title,
                    abstract=abstract,
                    full_text=full_text,
                    model_name=model_name,
                    selected_domains=selected_domains,
                    mode=mode
                )
                doc_res = {
                    "filename": filename,
                    "title": title,
                    "predicted_domain": pred["predicted_domain"],
                    "confidence": pred["confidence"],
                    "alternatives": pred["alternatives"],
                    "all_probabilities": pred.get("all_probabilities", {}),
                    "review_status": pred["review_status"],
                    "strategy_note": pred["strategy_note"]
                }

                dom = pred["predicted_domain"]
                domain_counts[dom] = domain_counts.get(dom, 0) + 1

                if pred["review_status"] == "Needs Review / Low Confidence":
                    uncertain_papers.append(filename)

            results.append(doc_res)

        # Batch summary statistics
        successfully_classified = sum(1 for r in results if not r["predicted_domain"].startswith("Unclassifiable"))
        batch_summary = {
            "total_papers": len(extracted_papers),
            "successfully_classified": successfully_classified,
            "domain_counts": domain_counts,
            "uncertain_papers": uncertain_papers,
            "uncertainty_rate": round(len(uncertain_papers) / max(len(extracted_papers), 1), 3),
            "model_used": model_name,
            "mode": mode
        }

        return results, batch_summary
