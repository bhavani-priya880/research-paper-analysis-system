"""
Configuration settings for the NLP-Based Multi-Domain Research Paper Analysis System.
Provides directory paths, domain mappings, model hyper-parameters, and inference thresholds.
"""

from pathlib import Path
import os

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_PAPERS_DIR = DATA_DIR / "sample_papers"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Kaggle dataset default path (as specified by user)
KAGGLE_INPUT_DIR = Path("/kaggle/input/datasets/sumitm004/arxiv-scientific-research-papers-dataset")
LOCAL_DATASET_CSV = DATA_DIR / "arXiv_scientific_dataset.csv"
SAMPLE_DATASET_CSV = DATA_DIR / "sample_arxiv_metadata.csv"

# Model artifact subdirectories
BASELINE_MODEL_DIR = MODELS_DIR / "baseline"
BILSTM_MODEL_DIR = MODELS_DIR / "bilstm"
SCIBERT_MODEL_DIR = MODELS_DIR / "scibert"

# Ensure directories exist
for p in [DATA_DIR, SAMPLE_PAPERS_DIR, MODELS_DIR, REPORTS_DIR, BASELINE_MODEL_DIR, BILSTM_MODEL_DIR, SCIBERT_MODEL_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Standardized Research Domains and arXiv category prefix mapping
# arXiv uses hierarchical classification codes, e.g., cs.CL, stat.ML, math.OC, quant-ph
ARXIV_PREFIX_TO_DOMAIN = {
    # Computer Science
    "cs": "Computer Science",
    # Mathematics
    "math": "Mathematics",
    # Physics domains
    "physics": "Physics",
    "quant-ph": "Physics",
    "cond-mat": "Physics",
    "astro-ph": "Physics",
    "hep-th": "Physics",
    "hep-ph": "Physics",
    "hep-lat": "Physics",
    "hep-ex": "Physics",
    "nucl-th": "Physics",
    "nucl-ex": "Physics",
    "gr-qc": "Physics",
    "nlin": "Physics",
    # Statistics
    "stat": "Statistics",
    # Quantitative Biology
    "q-bio": "Quantitative Biology",
    # Electrical Engineering and Systems Science
    "eess": "Electrical Engineering",
    # Economics and Quantitative Finance
    "econ": "Economics",
    "q-fin": "Quantitative Finance"
}

# Primary supported domains for multi-class classification
SUPPORTED_DOMAINS = [
    "Computer Science",
    "Mathematics",
    "Physics",
    "Statistics",
    "Quantitative Biology",
    "Electrical Engineering"
]

# Random Seed for Reproducibility
RANDOM_SEED = 42

# Confidence & Review Thresholds
# If top prediction probability is below this threshold, mark as 'Needs Review / Low Confidence'
UNCERTAINTY_THRESHOLD = 0.55

# PDF Processing Configuration
MAX_PDF_FILE_SIZE_MB = 50
SCANNED_PDF_CHAR_THRESHOLD = 80  # Fewer characters than this indicates a scanned/image-only PDF

# Model Hyperparameters
# 1. Baseline Model (TF-IDF + Logistic Regression / Linear SVM)
BASELINE_CONFIG = {
    "max_features": 10000,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95,
    "c_param": 1.0,
    "class_weight": "balanced"
}

# 2. Comparison Model: BiLSTM
BILSTM_CONFIG = {
    "vocab_size": 20000,
    "embedding_dim": 128,
    "hidden_dim": 128,
    "num_layers": 2,
    "bidirectional": True,
    "dropout": 0.3,
    "max_seq_len": 256,
    "batch_size": 32,
    "learning_rate": 0.003,
    "epochs": 15,
    "patience": 6
}

# 3. Proposed Main Model: SciBERT
SCIBERT_CONFIG = {
    "model_name": "allenai/scibert_scivocab_uncased",
    "fallback_model_name": "bert-base-uncased",
    "max_seq_len": 512,
    "chunk_size": 512,
    "chunk_overlap": 64,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "epochs": 3,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01
}
