# Models Directory

Contains trained model checkpoints, tokenizers, vocabulary mappings, and evaluation reports.

## Subdirectories
- `baseline/`:
  - `baseline_pipeline.joblib`: Serialized TF-IDF Vectorizer + Multinomial Logistic Regression pipeline.
  - `label_mapping.json`: Mapping from domain names to integer indices.
  - `baseline_metrics.json`: Evaluated test metrics (Macro F1, Accuracy, Precision, Recall, Confusion Matrix).
- `bilstm/`:
  - `bilstm_model.pt`: PyTorch weights and network architecture hyperparameters.
  - `tokenizer_vocab.json`: Vocabulary dictionary mapping tokens to IDs.
  - `bilstm_metrics.json`: Test split evaluation metrics and inference latency.
- `scibert/`:
  - `scibert_metrics.json`: Verified empirical evaluation metrics for `allenai/scibert_scivocab_uncased`.
  - Checkpoint weights and tokenizer configurations exported from Kaggle training.
