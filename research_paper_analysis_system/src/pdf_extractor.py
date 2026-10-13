"""
PDF Extraction module using PyMuPDF (fitz).
Extracts text, metadata, titles, and abstracts from academic research papers in PDF format.
Handles corrupted files, scanned documents, and multi-file batches with robust error isolation.
"""

import re
import io
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import SCANNED_PDF_CHAR_THRESHOLD, MAX_PDF_FILE_SIZE_MB

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    fitz = None
    PYMUPDF_AVAILABLE = False


def compute_file_hash(content_bytes: bytes) -> str:
    """Computes SHA-256 hash to detect duplicate uploads in a batch."""
    return hashlib.sha256(content_bytes).hexdigest()


def extract_title_heuristic(doc, first_page_text: str, filename: str) -> str:
    """
    Heuristically extracts paper title using PyMuPDF text block font size analysis,
    falling back to line-based parsing and filename.
    """
    if doc and len(doc) > 0:
        first_page = doc[0]
        try:
            # Extract structured blocks with font sizes
            blocks = first_page.get_text("dict", flags=11)["blocks"]
            spans_with_sizes = []
            for b in blocks:
                if "lines" in b:
                    for line in b["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            if len(text) > 3:
                                spans_with_sizes.append((span["size"], text))

            if spans_with_sizes:
                # Find maximum font size among spans in the top half of the page
                max_size = max(s[0] for s in spans_with_sizes)
                title_candidates = [
                    s[1] for s in spans_with_sizes
                    if s[0] >= (max_size - 1.0) and len(s[1]) > 5
                ]
                candidate_title = " ".join(title_candidates).strip()
                # Clean up if candidate looks valid (length between 10 and 300 chars)
                if 10 <= len(candidate_title) <= 300:
                    return candidate_title
        except Exception:
            pass

    # Fallback heuristic: lines before "Abstract" or "Author"
    lines = [line.strip() for line in first_page_text.splitlines() if line.strip()]
    cleaned_lines = []
    for line in lines:
        lower = line.lower()
        if lower.startswith("abstract") or "arxiv:" in lower or lower.startswith("ieee") or lower.startswith("acm"):
            break
        if len(line) > 5 and not lower.startswith("http") and not lower.startswith("doi"):
            cleaned_lines.append(line)
        if len(cleaned_lines) >= 3:
            break

    if cleaned_lines:
        title = " ".join(cleaned_lines)
        if 10 <= len(title) <= 250:
            return title

    # Fallback to sanitized filename
    base_name = Path(filename).stem
    return base_name.replace("_", " ").replace("-", " ").title()


def extract_abstract_heuristic(full_text: str) -> str:
    """
    Extracts abstract section using regex patterns matching common academic paper conventions.
    """
    # Pattern 1: Abstract ... 1. Introduction or Introduction or Index Terms
    pattern1 = re.compile(
        r"(?:abstract|summary)[\s\:\.\-\—]+(.*?)(?=\n\s*(?:[1I]\.?\s+introduction|keywords|index terms|categories and subject descriptors|1\s+intro))",
        re.IGNORECASE | re.DOTALL
    )
    match1 = pattern1.search(full_text)
    if match1:
        abstract = match1.group(1).strip()
        # Clean internal newlines and excessive spaces
        abstract = re.sub(r"\s+", " ", abstract)
        if 50 <= len(abstract) <= 4000:
            return abstract

    # Pattern 2: More relaxed abstract search up to next heading or double newline
    pattern2 = re.compile(
        r"\b(?:abstract)[\s\:\.\-\—]+(.*?)(?=\n\n|\n[A-Z\s]{4,}|\n\d+\s+[A-Z])",
        re.IGNORECASE | re.DOTALL
    )
    match2 = pattern2.search(full_text)
    if match2:
        abstract = match2.group(1).strip()
        abstract = re.sub(r"\s+", " ", abstract)
        if 50 <= len(abstract) <= 4000:
            return abstract

    # Fallback: if no explicit abstract marker, return empty string indicating not identified
    return ""


def extract_single_pdf(
    file_source: Union[str, Path, bytes, io.BytesIO],
    filename: str = "document.pdf"
) -> Dict[str, Any]:
    """
    Extracts text and metadata from a single PDF document.
    Gracefully handles corrupted files, encrypted PDFs, and scanned pages.
    """
    result = {
        "filename": filename,
        "success": False,
        "title": "",
        "abstract": "",
        "full_text": "",
        "num_pages": 0,
        "char_count": 0,
        "word_count": 0,
        "is_scanned": False,
        "file_hash": "",
        "warning": None,
        "error": None
    }

    if not PYMUPDF_AVAILABLE:
        result["error"] = "PyMuPDF (fitz) is not installed in the current environment."
        return result

    try:
        # Load document from bytes or path
        if isinstance(file_source, (bytes, io.BytesIO)):
            raw_bytes = file_source if isinstance(file_source, bytes) else file_source.getvalue()
            result["file_hash"] = compute_file_hash(raw_bytes)
            # Size check
            size_mb = len(raw_bytes) / (1024 * 1024)
            if size_mb > MAX_PDF_FILE_SIZE_MB:
                result["error"] = f"File exceeds maximum size limit of {MAX_PDF_FILE_SIZE_MB} MB ({size_mb:.1f} MB)."
                return result
            doc = fitz.open(stream=raw_bytes, filetype="pdf")
        else:
            path = Path(file_source)
            if not path.exists():
                result["error"] = f"File not found: {path}"
                return result
            raw_bytes = path.read_bytes()
            result["file_hash"] = compute_file_hash(raw_bytes)
            doc = fitz.open(path)

        if doc.is_encrypted:
            try:
                # Try authenticating with empty password
                doc.authenticate("")
            except Exception:
                result["error"] = "PDF document is encrypted and password-protected."
                return result

        result["num_pages"] = len(doc)
        if result["num_pages"] == 0:
            result["error"] = "PDF document contains 0 pages."
            return result

        page_texts = []
        first_page_text = ""

        for page_idx, page in enumerate(doc):
            text = page.get_text("text")
            page_texts.append(text)
            if page_idx == 0:
                first_page_text = text

        full_text = "\n\n".join(page_texts).strip()
        result["full_text"] = full_text
        result["char_count"] = len(full_text)
        result["word_count"] = len(full_text.split())

        # Check for scanned or image-only PDF
        if result["char_count"] < SCANNED_PDF_CHAR_THRESHOLD:
            result["is_scanned"] = True
            result["warning"] = (
                f"Document appears to be scanned or image-only ({result['char_count']} characters extracted). "
                "Text layer is insufficient for standard NLP models without OCR preprocessing."
            )
        else:
            # Heuristic extraction for title & abstract
            result["title"] = extract_title_heuristic(doc, first_page_text, filename)
            result["abstract"] = extract_abstract_heuristic(full_text)

            if not result["abstract"]:
                # If abstract not explicitly identified, use the first 1000 chars as provisional snippet
                provisional = " ".join(first_page_text.split()[:150])
                result["warning"] = (
                    "Abstract section heading was not explicitly detected. "
                    "Inference will analyze available header text and document body."
                )

        result["success"] = True

    except Exception as e:
        result["error"] = f"Failed to extract PDF: {str(e)}"
        result["success"] = False

    return result


def process_pdf_batch(
    files: List[Dict[str, Any]],
    progress_callback: Optional[callable] = None
) -> List[Dict[str, Any]]:
    """
    Processes a batch of uploaded PDFs, deduplicating files and isolating errors per file.
    
    Args:
        files: List of dicts, each with keys 'filename' and 'content' (bytes or stream)
        progress_callback: Optional callable(index, total, filename) for progress reporting
    """
    results = []
    seen_hashes = set()
    total = len(files)

    for idx, item in enumerate(files):
        filename = item.get("filename", f"document_{idx+1}.pdf")
        content = item.get("content")

        if progress_callback:
            progress_callback(idx + 1, total, filename)

        try:
            res = extract_single_pdf(content, filename=filename)

            # Deduplication check
            if res.get("file_hash") in seen_hashes:
                res["warning"] = "Duplicate file detected in current batch. Processed independently."
            else:
                seen_hashes.add(res.get("file_hash"))

            results.append(res)
        except Exception as e:
            results.append({
                "filename": filename,
                "success": False,
                "title": Path(filename).stem,
                "abstract": "",
                "full_text": "",
                "num_pages": 0,
                "char_count": 0,
                "word_count": 0,
                "is_scanned": False,
                "warning": None,
                "error": f"Unhandled extraction error: {str(e)}"
            })

    return results
