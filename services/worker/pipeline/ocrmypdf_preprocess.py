"""OCRmyPDF preprocessing: adds searchable text layer to scanned PDFs.

After preprocessing, the detector classifies the PDF as pdf_digital,
and PyMuPDF extracts text in milliseconds. 10-50x faster than Docling.
"""

import logging
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger("gutenberg.ocrmypdf")


def ocrmypdf_available() -> bool:
    """Check if ocrmypdf binary is available."""
    return shutil.which("ocrmypdf") is not None


def preprocess(input_path: Path, output_dir: Path) -> Path:
    """Add searchable text layer to a PDF using OCRmyPDF + Tesseract.

    Returns path to the preprocessed PDF in output_dir.
    Raises FileNotFoundError if ocrmypdf is not installed.
    Raises RuntimeError if OCR processing fails.
    """
    if not ocrmypdf_available():
        raise FileNotFoundError(
            "ocrmypdf not found. Install with: apt-get install ocrmypdf tesseract-ocr"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / input_path.name

    log.info(f"OCRmyPDF preprocessing: {input_path.name}")
    result = subprocess.run(
        [
            "ocrmypdf",
            "--force-ocr",         # OCR all pages (for scanned PDFs)
            "--optimize", "1",     # Light optimization
            "--jobs", "4",         # Parallel page processing
            str(input_path),
            str(output_path),
        ],
        capture_output=True,
        text=True,
        timeout=600,  # 10 min timeout per document
    )

    if result.returncode != 0:
        log.error(f"OCRmyPDF failed: {result.stderr[:500]}")
        raise RuntimeError(f"OCRmyPDF failed on {input_path.name}: {result.stderr[:200]}")

    log.info(f"OCRmyPDF done: {input_path.name} → {output_path}")
    return output_path
