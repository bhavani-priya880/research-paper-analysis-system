"""
Data Loader module for the Research Paper Analysis System.
Provides recursive directory discovery, automatic schema detection,
label mapping, data sanitization, and class distribution diagnostics.
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    KAGGLE_INPUT_DIR,
    LOCAL_DATASET_CSV,
    SAMPLE_DATASET_CSV,
    ARXIV_PREFIX_TO_DOMAIN,
    SUPPORTED_DOMAINS
)


def inspect_dataset_directory(dir_path: Path) -> Dict[str, Any]:
    """
    Recursively inspects a dataset directory and identifies its actual structure,
    file formats, sizes, and file list without making assumptions about layout.
    """
    path = Path(dir_path)
    report = {
        "directory": str(path),
        "exists": path.exists(),
        "files_found": [],
        "supported_data_files": [],
        "total_files": 0
    }

    if not path.exists():
        return report

    supported_exts = {".csv", ".tsv", ".parquet", ".json", ".jsonl"}

    for root, _, files in os.walk(path):
        for file in files:
            file_path = Path(root) / file
            rel_path = file_path.relative_to(path)
            size_mb = round(file_path.stat().st_size / (1024 * 1024), 2)
            file_info = {
                "relative_path": str(rel_path),
                "absolute_path": str(file_path),
                "extension": file_path.suffix.lower(),
                "size_mb": size_mb
            }
            report["files_found"].append(file_info)
            if file_path.suffix.lower() in supported_exts:
                report["supported_data_files"].append(file_info)

    report["total_files"] = len(report["files_found"])
    return report


def detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Dynamically identifies column names for title, summary/abstract, category/labels, and ID.
    Avoids hardcoding column assumptions.
    """
    columns = [str(c).strip().lower() for c in df.columns]
    col_mapping = {
        "id": None,
        "title": None,
        "summary": None,
        "category": None,
        "category_code": None
    }

    # Match ID
    id_candidates = ["id", "paper_id", "arxiv_id", "article_id", "doi"]
    for cand in id_candidates:
        for orig, lower in zip(df.columns, columns):
            if lower == cand:
                col_mapping["id"] = orig
                break
        if col_mapping["id"]:
            break

    # Match Title
    title_candidates = ["title", "paper_title", "article_title", "headline"]
    for cand in title_candidates:
        for orig, lower in zip(df.columns, columns):
            if lower == cand:
                col_mapping["title"] = orig
                break
        if col_mapping["title"]:
            break

    # Match Summary/Abstract
    summary_candidates = ["summary", "abstract", "description", "paper_summary", "text", "body"]
    for cand in summary_candidates:
        for orig, lower in zip(df.columns, columns):
            if lower == cand:
                col_mapping["summary"] = orig
                break
        if col_mapping["summary"]:
            break

    # Match Category Code
    code_candidates = ["category_code", "category_id", "primary_category", "subject_code", "categories"]
    for cand in code_candidates:
        for orig, lower in zip(df.columns, columns):
            if lower == cand:
                col_mapping["category_code"] = orig
                break
        if col_mapping["category_code"]:
            break

    # Match Category Name
    cat_candidates = ["category", "category_name", "subject", "domain", "topic", "field"]
    for cand in cat_candidates:
        for orig, lower in zip(df.columns, columns):
            if lower == cand and orig != col_mapping.get("category_code"):
                col_mapping["category"] = orig
                break
        if col_mapping["category"]:
            break

    return col_mapping


def map_arxiv_category_to_domain(category_val: str, category_code_val: Optional[str] = None) -> Optional[str]:
    """
    Defensibly maps arXiv category code or category description to standardized broad domain.
    Preserves academic validity by mapping only recognized arXiv prefixes.
    """
    candidate_code = ""
    if category_code_val and pd.notna(category_code_val):
        candidate_code = str(category_code_val).strip()
    elif category_val and pd.notna(category_val):
        candidate_code = str(category_val).strip()

    if not candidate_code:
        return None

    # Handle multiple categories separated by space or semicolon (e.g. "cs.CV cs.AI")
    first_code = candidate_code.split()[0].split(";")[0].strip()

    # Extract prefix (e.g., "cs" from "cs.LG", "quant-ph" from "quant-ph", "math" from "math.PR")
    prefix = first_code.split(".")[0].lower() if "." in first_code else first_code.lower()

    if prefix in ARXIV_PREFIX_TO_DOMAIN:
        return ARXIV_PREFIX_TO_DOMAIN[prefix]

    # Fallback checking full name matching
    category_lower = str(category_val).lower() if pd.notna(category_val) else ""
    if any(k in category_lower for k in ["computer science", "artificial intelligence", "machine learning", "computation and language", "computer vision"]):
        return "Computer Science"
    if any(k in category_lower for k in ["mathematics", "algebra", "analysis", "geometry", "topology", "optimization"]):
        return "Mathematics"
    if any(k in category_lower for k in ["physics", "quantum", "astrophysics", "condensed matter", "high energy"]):
        return "Physics"
    if any(k in category_lower for k in ["statistics", "methodology"]):
        return "Statistics"
    if any(k in category_lower for k in ["biology", "genomics", "neuroscience", "molecular"]):
        return "Quantitative Biology"
    if any(k in category_lower for k in ["electrical engineering", "signal processing", "systems and control"]):
        return "Electrical Engineering"

    return None


def load_dataset(
    file_path: Optional[Path] = None,
    target_domains: Optional[List[str]] = None,
    max_samples: Optional[int] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads, sanitizes, and prepares arXiv dataset from Kaggle or local paths.
    
    Returns:
        Tuple of (cleaned DataFrame, metadata dictionary)
    """
    resolved_path = None
    inspected_info = None

    # Determine data source
    candidate_paths = []
    if file_path:
        candidate_paths.append(Path(file_path))

    # Kaggle input paths
    candidate_paths.append(KAGGLE_INPUT_DIR)
    candidate_paths.append(Path("/kaggle/input/arxiv-scientific-research-papers-dataset"))
    candidate_paths.append(Path("/kaggle/input/datasets/sumitm004/arxiv-scientific-research-papers-dataset/arXiv_scientific_dataset.csv"))

    # Local paths
    candidate_paths.append(LOCAL_DATASET_CSV)
    candidate_paths.append(SAMPLE_DATASET_CSV)

    for p in candidate_paths:
        if p.is_dir():
            insp = inspect_dataset_directory(p)
            if insp["supported_data_files"]:
                resolved_path = Path(insp["supported_data_files"][0]["absolute_path"])
                inspected_info = insp
                break
        elif p.is_file() and p.exists():
            resolved_path = p
            break

    if not resolved_path or not resolved_path.exists():
        # Generate or return empty structure with informative status
        return pd.DataFrame(), {
            "status": "not_found",
            "message": f"Dataset file not found at inspected paths. Searched {len(candidate_paths)} candidate paths.",
            "searched_paths": [str(p) for p in candidate_paths]
        }

    # Load based on extension
    suffix = resolved_path.suffix.lower()
    if suffix in [".csv", ".tsv"]:
        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(resolved_path, sep=sep, low_memory=False)
    elif suffix == ".parquet":
        df = pd.read_parquet(resolved_path)
    elif suffix in [".json", ".jsonl"]:
        lines = suffix == ".jsonl"
        df = pd.read_json(resolved_path, lines=lines)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    total_raw_records = len(df)
    col_mapping = detect_columns(df)

    # Validate essential columns
    title_col = col_mapping["title"]
    summary_col = col_mapping["summary"]
    cat_col = col_mapping["category"]
    code_col = col_mapping["category_code"]

    if not title_col or not summary_col:
        return pd.DataFrame(), {
            "status": "schema_error",
            "message": f"Could not locate essential 'title' or 'summary/abstract' columns in {resolved_path.name}. Detected: {col_mapping}",
            "columns": list(df.columns)
        }

    # Standardize columns
    standard_df = pd.DataFrame()
    standard_df["id"] = df[col_mapping["id"]].astype(str) if col_mapping["id"] else [f"paper_{i}" for i in range(len(df))]
    standard_df["title"] = df[title_col].astype(str)
    standard_df["summary"] = df[summary_col].astype(str)
    standard_df["raw_category"] = df[cat_col].astype(str) if cat_col else ""
    standard_df["raw_category_code"] = df[code_col].astype(str) if code_col else ""

    # Optional metadata columns if available
    for opt_col in ["authors", "first_author", "published_date", "updated_date"]:
        for orig in df.columns:
            if orig.lower() == opt_col:
                standard_df[opt_col] = df[orig]
                break

    # Cleaning: remove nulls, whitespaces, and unusable text
    standard_df["title"] = standard_df["title"].str.strip()
    standard_df["summary"] = standard_df["summary"].str.strip()

    valid_mask = (
        (standard_df["title"].str.len() > 5) &
        (standard_df["summary"].str.len() > 30) &
        (standard_df["title"].str.lower() != "nan") &
        (standard_df["summary"].str.lower() != "nan")
    )
    cleaned_df = standard_df[valid_mask].copy()

    # Drop duplicates by title or ID to prevent leakage
    cleaned_df = cleaned_df.drop_duplicates(subset=["title"]).reset_index(drop=True)

    # Map labels to standardized research domains
    def get_domain(row):
        return map_arxiv_category_to_domain(row["raw_category"], row["raw_category_code"])

    cleaned_df["domain"] = cleaned_df.apply(get_domain, axis=1)

    # Drop unmapped domains
    unmapped_count = cleaned_df["domain"].isna().sum()
    cleaned_df = cleaned_df.dropna(subset=["domain"]).reset_index(drop=True)

    # Filter by target domains if provided
    if target_domains:
        cleaned_df = cleaned_df[cleaned_df["domain"].isin(target_domains)].reset_index(drop=True)

    # Subsample if max_samples is set
    if max_samples and len(cleaned_df) > max_samples:
        # Stratified sampling where possible
        cleaned_df = cleaned_df.groupby("domain", group_keys=False).apply(
            lambda x: x.sample(min(len(x), int(np.ceil(max_samples * len(x) / len(cleaned_df)))), random_state=42)
        ).head(max_samples).reset_index(drop=True)

    # Compute diagnostics
    class_distribution = cleaned_df["domain"].value_counts().to_dict()
    total_retained = len(cleaned_df)
    
    imbalance_ratio = 1.0
    if class_distribution and min(class_distribution.values()) > 0:
        imbalance_ratio = round(max(class_distribution.values()) / min(class_distribution.values()), 2)

    metadata = {
        "status": "success",
        "source_path": str(resolved_path),
        "total_raw_records": total_raw_records,
        "total_retained": total_retained,
        "unmapped_records": int(unmapped_count),
        "detected_columns": col_mapping,
        "class_distribution": class_distribution,
        "class_imbalance_ratio": imbalance_ratio,
        "supported_domains": list(class_distribution.keys()),
        "inspected_info": inspected_info
    }

    return cleaned_df, metadata
