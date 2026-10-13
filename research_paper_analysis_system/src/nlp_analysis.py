"""
NLP Analysis module.
Provides actual text analytics computed directly on extracted papers:
- N-gram word frequency analysis
- WordCloud generation
- TF-IDF distinctive keyword extraction
- Document statistics and lexical richness
- Scientific tone and sentiment analysis with academic limitation caveats
"""

import re
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.text_preprocessing import clean_text_for_nlp, segment_sentences, SCIENTIFIC_STOPWORDS

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WordCloud = None
    WORDCLOUD_AVAILABLE = False


def compute_word_frequencies(
    text: str,
    top_n: int = 20,
    include_bigrams: bool = True
) -> List[Tuple[str, int]]:
    """
    Computes top-N unigram and bigram word frequencies from cleaned research text.
    """
    cleaned = clean_text_for_nlp(text, remove_stopwords=True, lemmatize=True)
    words = cleaned.split()

    if not words:
        return []

    # Unigram counts
    counts = Counter(words)

    # Bigram counts
    if include_bigrams and len(words) >= 2:
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        # Filter bigrams where words are identical or generic
        valid_bigrams = [
            b for b in bigrams
            if b.split()[0] != b.split()[1] and
            b.split()[0] not in SCIENTIFIC_STOPWORDS and
            b.split()[1] not in SCIENTIFIC_STOPWORDS
        ]
        counts.update(valid_bigrams)

    return counts.most_common(top_n)


def generate_wordcloud_image(
    text: str,
    width: int = 800,
    height: int = 400,
    background_color: str = "white",
    colormap: str = "viridis"
):
    """
    Generates an actual WordCloud image from extracted research text.
    Returns a PIL Image or None if no valid words.
    """
    cleaned = clean_text_for_nlp(text, remove_stopwords=True, lemmatize=True)
    if not cleaned or len(cleaned.split()) < 3:
        return None

    if WORDCLOUD_AVAILABLE and WordCloud:
        wc = WordCloud(
            width=width,
            height=height,
            background_color=background_color,
            colormap=colormap,
            max_words=100,
            stopwords=SCIENTIFIC_STOPWORDS,
            collocations=True,
            contour_width=1,
            contour_color="steelblue"
        )
        wc.generate(cleaned)
        return wc.to_image()

    return None


def compute_tfidf_keywords(
    documents: List[str],
    doc_titles: Optional[List[str]] = None,
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Computes TF-IDF distinctive keywords and importance weights across a collection of papers.
    Explains why terms are distinctive (high frequency in paper, penalized by commonness in corpus).
    
    Returns:
        List of dicts for each document with top keywords and weights.
    """
    if not documents:
        return []

    cleaned_docs = [clean_text_for_nlp(doc, remove_stopwords=True) for doc in documents]
    
    # Filter out empty docs
    non_empty_indices = [i for i, d in enumerate(cleaned_docs) if len(d.strip()) > 0]
    if not non_empty_indices:
        return []

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(cleaned_docs)
        feature_names = np.array(vectorizer.get_feature_names_out())
    except Exception:
        return []

    results = []
    for idx, doc_text in enumerate(documents):
        title = doc_titles[idx] if doc_titles and idx < len(doc_titles) else f"Document {idx+1}"
        row = tfidf_matrix[idx].toarray().flatten()

        top_indices = row.argsort()[::-1][:top_k]
        top_terms = [
            {"term": feature_names[i], "score": round(float(row[i]), 4)}
            for i in top_indices if row[i] > 0
        ]

        results.append({
            "title": title,
            "top_keywords": top_terms,
            "distinctive_count": len(top_terms)
        })

    return results


def compute_document_statistics(text: str, filename: str = "") -> Dict[str, Any]:
    """
    Computes document-level statistical metrics:
    word count, unique vocabulary, sentence count, lexical diversity, readability estimate.
    """
    words = text.split()
    total_words = len(words)
    total_chars = len(text)
    
    cleaned_words = [w.lower() for w in re.findall(r"\b[a-zA-Z]+\b", text)]
    unique_words = len(set(cleaned_words))
    
    # Lexical Diversity (Type-Token Ratio)
    lexical_diversity = round(unique_words / max(total_words, 1), 3)

    sentences = segment_sentences(text)
    total_sentences = len(sentences)
    avg_sentence_len = round(total_words / max(total_sentences, 1), 1)

    # Academic estimated reading time (approx 180 words per minute for dense academic prose)
    est_reading_time_min = round(total_words / 180.0, 1)

    return {
        "filename": filename,
        "total_characters": total_chars,
        "total_words": total_words,
        "unique_words": unique_words,
        "total_sentences": total_sentences,
        "avg_sentence_length_words": avg_sentence_len,
        "lexical_diversity_ttr": lexical_diversity,
        "estimated_reading_time_min": est_reading_time_min
    }


def analyze_scientific_sentiment(text: str) -> Dict[str, Any]:
    """
    Analyzes sentiment and epistemic tone of scientific research papers.
    Includes explicit academic caveats regarding scientific objectivity and technical terminology.
    """
    lower = text.lower()
    words = re.findall(r"\b[a-z]+\b", lower)

    # Scientific positive evaluative terms (demonstrating efficacy, novelty, validation)
    positive_markers = {
        "significant", "superior", "outperform", "outperforms", "effective", "efficient", "novel",
        "innovative", "accurate", "robust", "substantially", "consistently", "promising", "breakthrough",
        "optimal", "seamless", "verified", "reliable", "advantage", "scalable", "exceptional"
    }

    # Scientific limitation / critical markers (acknowledging constraints, errors, vulnerability)
    limitation_markers = {
        "limitation", "limitations", "drawback", "drawbacks", "vulnerable", "vulnerability", "error",
        "errors", "bottleneck", "degradation", "underperform", "insufficient", "challenging", "unsolved",
        "failure", "trade-off", "constrained", "bias", "costly", "unstable", "sensitive"
    }

    # Epistemic hedging markers (academic modesty and probabilistic claims)
    hedge_markers = {
        "suggest", "suggests", "indicate", "indicates", "potential", "potentially", "likely",
        "plausible", "approximately", "partially", "preliminary", "hypothesize", "apparent"
    }

    pos_count = sum(1 for w in words if w in positive_markers)
    lim_count = sum(1 for w in words if w in limitation_markers)
    hedge_count = sum(1 for w in words if w in hedge_markers)

    total_eval = pos_count + lim_count
    if total_eval > 0:
        polarity = round((pos_count - lim_count) / total_eval, 3)
    else:
        polarity = 0.0

    subjectivity = round(min(1.0, (pos_count + lim_count + hedge_count) / max(len(words) * 0.05, 1)), 3)

    if polarity > 0.2:
        tone_label = "Positive / Validating Efficacy"
    elif polarity < -0.2:
        tone_label = "Critical / Limitation-Focused"
    else:
        tone_label = "Objective / Neutral Scientific Discourse"

    return {
        "polarity": polarity,
        "subjectivity": subjectivity,
        "tone_label": tone_label,
        "positive_terms_found": pos_count,
        "limitation_terms_found": lim_count,
        "hedging_terms_found": hedge_count,
        "academic_disclaimer": (
            "Academic Note: Research papers naturally adhere to objective and hedged discourse. "
            "A low polarity score reflects scientific rigor, objectivity, and peer-review neutrality, "
            "not negative sentiment. Scientific terms (e.g., 'loss', 'regression', 'decomposition') "
            "are domain-specific constructs, not negative affect."
        )
    }
