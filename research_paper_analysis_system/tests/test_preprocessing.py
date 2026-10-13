"""
Unit tests for text preprocessing pipeline.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.text_preprocessing import (
    clean_pdf_artifacts,
    clean_text_for_nlp,
    prepare_for_scibert,
    BiLSTMTokenizer,
    segment_sentences
)


def test_clean_pdf_artifacts():
    sample_raw = "This is a repre-\nsentation with citations [1, 2] and (Smith et al., 2020) and https://arxiv.org/abs/1234."
    cleaned = clean_pdf_artifacts(sample_raw)

    assert "representation" in cleaned, "Hyphenated word was not rejoined"
    assert "[1, 2]" not in cleaned, "Citation bracket was not removed"
    assert "https://" not in cleaned, "URL was not removed"


def test_scientific_term_preservation():
    sample_text = "We propose an AI framework using CNN and BiLSTM with p-value < 0.05 on RNA-seq data."
    cleaned_nlp = clean_text_for_nlp(sample_text)

    # Key acronyms should be preserved in lowercased forms
    assert "ai" in cleaned_nlp.split()
    assert "cnn" in cleaned_nlp.split()
    assert "bilstm" in cleaned_nlp.split()


def test_prepare_for_scibert():
    sample_text = "Deep residual learning for ImageNet. In Section 3, we analyze Eq. (4)."
    scibert_text = prepare_for_scibert(sample_text)

    # SciBERT text should preserve upper case and punctuation
    assert "ImageNet" in scibert_text
    assert "Section" in scibert_text


def test_bilstm_tokenizer():
    corpus = [
        "Convolutional neural networks for computer vision",
        "Deep transformers for natural language processing",
        "Reinforcement learning in artificial intelligence"
    ]
    tok = BiLSTMTokenizer(max_vocab_size=100, min_freq=1)
    tok.build_vocab(corpus)

    assert len(tok.word2idx) > 5
    assert tok.pad_token in tok.word2idx
    assert tok.unk_token in tok.word2idx

    encoded = tok.encode("neural networks for deep learning", max_seq_len=10)
    assert len(encoded) == 10, "Encoded sequence was not padded/truncated to max_seq_len"
    assert encoded[-1] == tok.pad_idx, "Sequence padding index mismatch"


if __name__ == "__main__":
    test_clean_pdf_artifacts()
    test_scientific_term_preservation()
    test_prepare_for_scibert()
    test_bilstm_tokenizer()
    print("All preprocessing unit tests passed!")
