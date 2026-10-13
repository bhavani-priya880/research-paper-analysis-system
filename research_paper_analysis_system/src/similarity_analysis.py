"""
Document Similarity Analysis module.
Calculates pairwise TF-IDF cosine similarity between uploaded research papers,
identifies topical clusters, and documents the distinction between lexical similarity and domain classification.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.text_preprocessing import clean_text_for_nlp


def compute_pairwise_similarity(
    documents: List[str],
    doc_titles: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Computes pairwise TF-IDF cosine similarity matrix for a batch of documents.
    Identifies high-similarity paper pairs and provides methodological explanations.
    """
    if len(documents) < 2:
        return {
            "matrix": [[1.0]],
            "titles": doc_titles if doc_titles else ["Paper 1"],
            "top_similar_pairs": [],
            "mean_similarity": 1.0,
            "explanation": "At least 2 documents are required for pairwise similarity computation."
        }

    cleaned_docs = [clean_text_for_nlp(d, remove_stopwords=True) for d in documents]
    
    # Handle empty texts gracefully
    cleaned_docs = [d if len(d.strip()) > 0 else "empty document text" for d in cleaned_docs]

    titles = doc_titles if doc_titles and len(doc_titles) == len(documents) else [
        f"Paper {i+1}" for i in range(len(documents))
    ]

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    tfidf_matrix = vectorizer.fit_transform(cleaned_docs)
    sim_matrix = cosine_similarity(tfidf_matrix)

    # Find highest similarity pairs (excluding diagonal self-similarity)
    n = len(documents)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            score = float(sim_matrix[i, j])
            pairs.append({
                "paper_a": titles[i],
                "paper_b": titles[j],
                "similarity_score": round(score, 4),
                "relationship": (
                    "High Overlap (Shared Subfield/Methods)" if score >= 0.50 else
                    "Moderate Topical Overlap" if score >= 0.25 else
                    "Low Overlap (Distinct Methodologies)"
                )
            })

    pairs.sort(key=lambda x: x["similarity_score"], reverse=True)

    # Compute off-diagonal average similarity
    off_diagonal = [p["similarity_score"] for p in pairs]
    mean_sim = round(float(np.mean(off_diagonal)), 4) if off_diagonal else 0.0

    return {
        "matrix": sim_matrix.round(4).tolist(),
        "titles": titles,
        "top_similar_pairs": pairs[:10],
        "mean_similarity": mean_sim,
        "methodological_note": (
            "Academic Distinction: Cosine similarity measures lexical and topical overlap between "
            "document representations in vector space. High similarity indicates shared terminology, "
            "experimental protocols, or mathematical tools (e.g. Graph Theory or Stochastic Optimization). "
            "However, similarity is NOT identical to domain classification: an optimization paper in Mathematics "
            "and an optimization paper in Machine Learning may exhibit high cosine similarity while legitimately "
            "belonging to distinct primary academic domains."
        )
    }
