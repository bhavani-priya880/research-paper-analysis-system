# Kaggle Training & Model Export Guide

This directory provides scripts and instructions to train the **Baseline (TF-IDF + Logistic Regression)**, **BiLSTM**, and **SciBERT (`allenai/scibert_scivocab_uncased`)** models using Kaggle's free GPU accelerators.

---

## 1. Setting Up the Kaggle Environment

1. **Create a New Notebook** on [Kaggle](https://www.kaggle.com/).
2. **Attach the Dataset**:
   - In the right-hand panel, click **+ Add Data**.
   - Search for: `sumitm004/arxiv-scientific-research-papers-dataset`.
   - Click **Add**. The dataset will be mounted at:
     ```
     /kaggle/input/datasets/sumitm004/arxiv-scientific-research-papers-dataset/
     ```
3. **Enable GPU Accelerator**:
   - In the notebook settings (three dots in upper right or Settings panel), select **Accelerator**: **GPU P100** or **GPU T4 x2**.
   - Turn **Internet** ON (required for downloading Hugging Face SciBERT weights).

---

## 2. Running Training in Kaggle

You can either clone this project repository into Kaggle or copy `train_kaggle.py` and the `src/` modules into your notebook.

### In a Notebook Cell:
```python
# 1. Install necessary dependencies
!pip install -q pymupdf wordcloud plotly transformers torch datasets

# 2. Run the Kaggle trainer
!python /kaggle/working/research_paper_analysis_system/kaggle/train_kaggle.py
```

### What `train_kaggle.py` Does:
1. **Recursively Inspects** `/kaggle/input/` to detect `arXiv_scientific_dataset.csv`.
2. **Cleans & Sanitizes** the dataset (drops duplicates, removes null abstracts, handles formatting).
3. **Maps arXiv Category Codes** (`cs.*`, `math.*`, `quant-ph`, etc.) to defensible research domains.
4. **Performs Stratified Splits** (70% train, 15% validation, 15% test) with verified title deduplication to prevent cross-split leakage.
5. **Trains Baseline**: TF-IDF + balanced Logistic Regression.
6. **Trains BiLSTM**: Bidirectional LSTM with class-weighted CrossEntropyLoss.
7. **Fine-tunes SciBERT**: `allenai/scibert_scivocab_uncased` using AdamW, linear learning rate schedule with warmup, and sliding-window chunk aggregation.
8. **Evaluates All Models** on the identical held-out test split, reporting:
   - Accuracy
   - Macro Precision, Recall, and F1-score
   - Per-class F1-scores
   - Confusion matrices
   - Inference latency
9. **Packages Outputs**: All checkpoints, tokenizers, label encoders, and evaluation reports are saved to `/kaggle/working/models/` and zipped to `/kaggle/working/trained_models_bundle.zip`.

---

## 3. Exporting Checkpoints to Local Application

1. In the Kaggle notebook right sidebar, expand the **Output** section.
2. Locate `/kaggle/working/trained_models_bundle.zip`.
3. Click the three dots next to the file and select **Download**.
4. In your local project directory, unzip the bundle:
   ```bash
   unzip trained_models_bundle.zip -d .
   ```
5. Launch the local web dashboard:
   ```bash
   streamlit run app.py
   ```
The local application will automatically detect the fine-tuned SciBERT weights, BiLSTM checkpoints, and baseline models!
