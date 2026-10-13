"""
Unit tests for PDF extraction module.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.pdf_extractor import extract_single_pdf, process_pdf_batch, compute_file_hash
from config import SAMPLE_PAPERS_DIR


def test_sample_pdf_extraction():
    sample_files = list(SAMPLE_PAPERS_DIR.glob("*.pdf"))
    assert len(sample_files) > 0, "No sample PDF files found in data/sample_papers/"

    pdf_path = sample_files[0]
    result = extract_single_pdf(pdf_path, filename=pdf_path.name)

    assert result["success"] is True, f"Extraction failed: {result.get('error')}"
    assert result["char_count"] > 100, "Character count unexpectedly low"
    assert result["word_count"] > 20, "Word count unexpectedly low"
    assert result["is_scanned"] is False, "Valid PDF was incorrectly flagged as scanned"
    assert len(result["title"]) > 5, "Extracted title is too short"
    assert result["file_hash"] != "", "File hash was not computed"


def test_batch_extraction_and_deduplication():
    sample_files = list(SAMPLE_PAPERS_DIR.glob("*.pdf"))
    assert len(sample_files) >= 2, "Need at least 2 sample PDFs for batch test"

    # Pass two files plus one duplicate
    items = [
        {"filename": sample_files[0].name, "content": sample_files[0].read_bytes()},
        {"filename": sample_files[1].name, "content": sample_files[1].read_bytes()},
        {"filename": "duplicate_copy.pdf", "content": sample_files[0].read_bytes()}  # exact same bytes
    ]

    results = process_pdf_batch(items)
    assert len(results) == 3, "Batch did not return results for all items"
    assert results[0]["success"] is True
    assert results[1]["success"] is True
    # The third duplicate should have a duplicate warning
    assert results[2]["warning"] is not None and "duplicate" in results[2]["warning"].lower()


if __name__ == "__main__":
    test_sample_pdf_extraction()
    test_batch_extraction_and_deduplication()
    print("All PDF extraction unit tests passed!")
