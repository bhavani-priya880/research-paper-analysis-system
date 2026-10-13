"""
Unit tests for domain prediction and inference engine.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.prediction import PaperDomainPredictor
from config import SUPPORTED_DOMAINS


def test_predictor_initialization_and_inference():
    predictor = PaperDomainPredictor()
    available_models = predictor.get_available_model_names()
    assert len(available_models) > 0, "No models were loaded by PaperDomainPredictor"

    # Test single paper prediction
    title = "Attention Is All You Need for Transformer Networks"
    abstract = "We propose the Transformer architecture based on attention mechanisms, discarding recurrence."
    full_text = f"{title}. {abstract}. Section 1. Introduction to Natural Language Processing."

    pred = predictor.predict_single_paper(
        title=title,
        abstract=abstract,
        full_text=full_text,
        model_name=available_models[0],
        mode="unrestricted"
    )

    assert "predicted_domain" in pred, "Prediction missing 'predicted_domain'"
    assert pred["predicted_domain"] in SUPPORTED_DOMAINS, f"Domain {pred['predicted_domain']} not in supported domains"
    assert "confidence" in pred, "Prediction missing 'confidence'"
    assert 0.0 <= pred["confidence"] <= 1.0, "Confidence score out of range [0, 1]"
    assert "review_status" in pred, "Prediction missing 'review_status'"
    assert "alternatives" in pred, "Prediction missing 'alternatives'"


def test_batch_prediction():
    predictor = PaperDomainPredictor()
    available = predictor.get_available_model_names()[0]

    sample_batch = [
        {
            "filename": "test_cs.pdf",
            "title": "Deep Convolutional Neural Networks for ImageNet Classification",
            "abstract": "We trained a large deep convolutional neural network on ImageNet.",
            "full_text": "We trained a deep convolutional network for computer vision.",
            "success": True,
            "is_scanned": False
        },
        {
            "filename": "test_math.pdf",
            "title": "Stochastic Gradient Descent in Non-Convex Hilbert Spaces",
            "abstract": "We analyze convergence bounds for stochastic optimization.",
            "full_text": "We analyze stochastic gradient descent in Hilbert spaces.",
            "success": True,
            "is_scanned": False
        }
    ]

    results, summary = predictor.predict_batch(sample_batch, model_name=available)

    assert len(results) == 2, "Batch prediction count mismatch"
    assert summary["total_papers"] == 2
    assert summary["successfully_classified"] == 2
    assert "domain_counts" in summary


if __name__ == "__main__":
    test_predictor_initialization_and_inference()
    test_batch_prediction()
    print("All prediction unit tests passed!")
