"""
NLP Text Preprocessing Pipeline.
Provides distinct representations for:
1. Conventional NLP analysis (word frequency, TF-IDF, similarity)
2. SciBERT Transformer input (syntactically intact, ligature & artifact cleaned)
3. BiLSTM sequence encoding (tokenized, vocab indexed, padded)
Preserves scientific terms, handles citations, equations, and extraction artifacts.
"""

import re
import string
from typing import List, Dict, Optional, Tuple, Set


# Common scientific stopwords that dilute term analysis but don't define domains
SCIENTIFIC_STOPWORDS: Set[str] = {
    # Standard English stopwords
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can",
    "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't",
    "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have",
    "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him",
    "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't",
    "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor",
    "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out",
    "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through",
    "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've",
    "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who",
    "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll",
    "you're", "you've", "your", "yours", "yourself", "yourselves",
    # Academic publication boilerplate
    "et", "al", "paper", "propose", "proposed", "presents", "presented", "study", "studies", "results",
    "approach", "method", "methods", "using", "used", "also", "based", "show", "shown", "shows", "well",
    "two", "one", "three", "first", "second", "new", "via", "table", "figure", "fig", "section", "demonstrate",
    "demonstrates", "furthermore", "moreover", "however", "therefore", "thus", "either", "neither", "respectively"
}

# Scientific preservation patterns: terms, symbols, acronyms that should NOT be split
# e.g. "p-value", "f1-score", "k-means", "l1-norm", "o(n)", "c++", "rna-seq"
PRESERVED_TERMS = {
    "ai", "ml", "nlp", "cv", "cnn", "rnn", "lstm", "bilstm", "bert", "scibert", "gpt", "svm", "knn",
    "pca", "gan", "rl", "dna", "rna", "mrna", "crispr", "mri", "ct", "eeg", "ecg", "pde", "ode", "qft",
    "rgb", "qkd", "fpga", "gpu", "tpu", "cpu", "io", "api", "iot", "f1", "auc", "roc"
}


def clean_pdf_artifacts(text: str) -> str:
    """
    Removes extraction artifacts such as broken hyphenated words across line wraps,
    header/footer page numbers, and redundant whitespace.
    """
    if not text:
        return ""

    # Fix hyphenation at line breaks (e.g. "repre-\nsentation" -> "representation")
    cleaned = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)

    # Normalize unicode ligatures (ﬁ -> fi, ﬂ -> fl, etc.)
    ligatures = {
        "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl", "—": " - ", "–": " - "
    }
    for lig, rep in ligatures.items():
        cleaned = cleaned.replace(lig, rep)

    # Remove standard citation markers (e.g. [1], [2, 3], [12-15])
    cleaned = re.sub(r"\[\s*\d+(?:\s*[\,\-–]\s*\d+)*\s*\]", " ", cleaned)

    # Remove author-year citations like (Smith et al., 2021) or (Vaswani et al. 2017)
    cleaned = re.sub(r"\([A-Z][a-zA-Z\s]+et\s+al\.?,?\s*\d{4}\)", " ", cleaned)

    # Remove URLs, email addresses, and DOIs
    cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)
    cleaned = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " ", cleaned)
    cleaned = re.sub(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b", " ", cleaned)

    # Collapse excessive spaces and consecutive blank lines
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


def segment_sentences(text: str) -> List[str]:
    """
    Splits text into sentences while respecting common academic abbreviations (e.g., e.g., i.e., Fig., Eq.).
    """
    if not text:
        return []

    # Protect common academic abbreviations
    protected = text
    abbrevs = ["e.g.", "i.e.", "et al.", "Fig.", "Eq.", "Ref.", "vol.", "no.", "pp.", "vs."]
    for idx, abb in enumerate(abbrevs):
        protected = protected.replace(abb, f"__ABBR{idx}__")

    # Split on sentence boundaries
    raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", protected)

    restored = []
    for s in raw_sentences:
        sent = s.strip()
        for idx, abb in enumerate(abbrevs):
            sent = sent.replace(f"__ABBR{idx}__", abb)
        if len(sent) > 5:
            restored.append(sent)

    return restored


def simple_lemmatize(word: str) -> str:
    """
    Lightweight rule-based lemmatizer for academic terminology
    without requiring heavy external corpus downloads.
    """
    if word in PRESERVED_TERMS or len(word) <= 3:
        return word

    # Plural nouns
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ses") and len(word) > 4:
        return word[:-2]
    if word.endswith("xes") or word.endswith("ches") or word.endswith("shes"):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]

    # Verb endings
    if word.endswith("ing") and len(word) > 5:
        base = word[:-3]
        if len(base) > 2 and base[-1] == base[-2] and base[-1] not in "lsz":
            return base[:-1]  # running -> run
        return base
    if word.endswith("ed") and len(word) > 4:
        base = word[:-2]
        if len(base) > 2 and base[-1] == base[-2]:
            return base[:-1]  # mapped -> map
        return base

    return word


def clean_text_for_nlp(
    text: str,
    remove_stopwords: bool = True,
    lemmatize: bool = True,
    min_word_len: int = 2
) -> str:
    """
    Preprocesses text for conventional NLP analysis: word frequency, TF-IDF, similarity.
    Normalizes case, removes stopwords and punctuation, but preserves key scientific acronyms.
    """
    text = clean_pdf_artifacts(text)
    if not text:
        return ""

    # Convert to lowercase
    lower_text = text.lower()

    # Replace mathematical expressions and bracketed equations with space
    lower_text = re.sub(r"\$+[^\$]+\$+", " ", lower_text)
    lower_text = re.sub(r"\\\([^\)]+\\\)", " ", lower_text)
    lower_text = re.sub(r"\\\[[^\]]+\\\]", " ", lower_text)

    # Tokenize preserving hyphenated scientific terms (e.g. "cross-validation", "deep-learning")
    tokens = re.findall(r"\b[a-z0-9]+(?:-[a-z0-9]+)*\b", lower_text)

    processed_tokens = []
    for t in tokens:
        if len(t) < min_word_len:
            continue
        # Drop pure digits
        if t.isdigit():
            continue

        if remove_stopwords and t in SCIENTIFIC_STOPWORDS:
            continue

        token_to_add = simple_lemmatize(t) if lemmatize else t
        if remove_stopwords and token_to_add in SCIENTIFIC_STOPWORDS:
            continue

        processed_tokens.append(token_to_add)

    return " ".join(processed_tokens)


def prepare_for_scibert(text: str, max_chars: Optional[int] = None) -> str:
    """
    Prepares text for SciBERT Transformer inference.
    Preserves casing, punctuation, grammatical sentence structure, and scientific terminology.
    Cleans extraction noise and line breaks.
    """
    cleaned = clean_pdf_artifacts(text)
    # Join artificial single newlines within paragraphs into single space
    cleaned = re.sub(r"(?<!\n)\n(?!\n)", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if max_chars and len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rsplit(" ", 1)[0]

    return cleaned


class BiLSTMTokenizer:
    """
    Vocabulary builder and sequence encoder/decoder for BiLSTM.
    Manages <pad>, <unk>, frequency thresholding, and sequence padding.
    """
    def __init__(self, max_vocab_size: int = 20000, min_freq: int = 2):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.pad_idx = 0
        self.unk_idx = 1
        self.word2idx: Dict[str, int] = {self.pad_token: self.pad_idx, self.unk_token: self.unk_idx}
        self.idx2word: Dict[int, str] = {self.pad_idx: self.pad_token, self.unk_idx: self.unk_token}

    def build_vocab(self, texts: List[str]):
        """Builds vocabulary mapping from list of raw or preprocessed texts."""
        word_counts: Dict[str, int] = {}
        for text in texts:
            tokens = re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())
            for t in tokens:
                word_counts[t] = word_counts.get(t, 0) + 1

        # Sort words by frequency
        sorted_words = sorted(
            [w for w, c in word_counts.items() if c >= self.min_freq],
            key=lambda w: word_counts[w],
            reverse=True
        )

        for w in sorted_words:
            if len(self.word2idx) >= self.max_vocab_size:
                break
            idx = len(self.word2idx)
            self.word2idx[w] = idx
            self.idx2word[idx] = w

    def encode(self, text: str, max_seq_len: int = 256) -> List[int]:
        """Encodes text to integer token indices with padding/truncation."""
        tokens = re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())
        indices = [self.word2idx.get(t, self.unk_idx) for t in tokens[:max_seq_len]]

        # Pad if shorter
        if len(indices) < max_seq_len:
            indices += [self.pad_idx] * (max_seq_len - len(indices))

        return indices

    def to_dict(self) -> Dict:
        """Serializes tokenizer vocabulary to dictionary."""
        return {
            "word2idx": self.word2idx,
            "max_vocab_size": self.max_vocab_size,
            "min_freq": self.min_freq
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "BiLSTMTokenizer":
        """Reconstructs tokenizer from serialized dictionary."""
        inst = cls(max_vocab_size=data.get("max_vocab_size", 20000), min_freq=data.get("min_freq", 2))
        inst.word2idx = data.get("word2idx", {})
        inst.idx2word = {int(idx): word for word, idx in inst.word2idx.items()}
        return inst
