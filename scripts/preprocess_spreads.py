#!/usr/bin/env python3
"""Preprocess scanned book spreads: rotate + split into individual pages.

These PDFs contain two-page book spreads photographed at 90° rotation.
Each PDF page has one image showing two book pages side by side (after rotation).

Pipeline: rotate 90° CCW → split left/right → output as individual pages.

Usage:
    uv run scripts/preprocess_spreads.py data/processed/1966*.bak
    uv run scripts/preprocess_spreads.py --all
"""

import argparse
import logging
import sys
from pathlib import Path

import pymupdf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("preprocess")

PROBLEM_BOOKS = [
    "1966 Bergsonism - Deleuze, Gilles.pdf.bak",
    "1969 The Logic of Sense - Deleuze, Gilles.pdf.bak",
    "1977 Dialogues - Deleuze, Gilles.pdf.bak",
    "1988 The Fold Leibniz and the Baroque - Deleuze, Gilles.pdf.bak",
]

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "split_pages"


def process_spread_pdf(input_path: Path, output_path: Path, dpi: int = 300):
    """Rotate and split a spread-scanned PDF into individual pages."""
    from PIL import Image
    import io

    doc = pymupdf.open(str(input_path))
    out_doc = pymupdf.open()  # new empty PDF

    log.info(f"Processing {input_path.name}: {len(doc)} spread pages")

    for pg_idx in range(len(doc)):
        page = doc[pg_idx]

        # Render page to pixmap, then convert to PIL for easy manipulation
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

        # Rotate 90° CCW (the spreads are photographed rotated CW)
        img_rot = img.rotate(90, expand=True)

        w, h = img_rot.size
        # After rotation: two book pages are stacked top-to-bottom
        # Top half = left page, bottom half = right page
        mid = h // 2

        for half_idx, (y_start, y_end) in enumerate([(0, mid), (mid, h)]):
            crop = img_rot.crop((0, y_start, w, y_end))

            # Convert PIL image to JPEG for smaller output
            buf = io.BytesIO()
            crop.save(buf, format="JPEG", quality=85)
            img_bytes = buf.getvalue()

            # Create new page
            page_w_pts = crop.width * 72 / dpi
            page_h_pts = crop.height * 72 / dpi
            new_page = out_doc.new_page(width=page_w_pts, height=page_h_pts)
            new_page.insert_image(
                pymupdf.Rect(0, 0, page_w_pts, page_h_pts),
                stream=img_bytes,
            )

        if (pg_idx + 1) % 10 == 0:
            log.info(f"  Processed {pg_idx + 1}/{len(doc)} spreads → {(pg_idx + 1) * 2} pages")

    out_doc.save(str(output_path))
    total_pages = len(out_doc)
    out_doc.close()
    doc.close()

    log.info(f"  Output: {output_path.name} ({total_pages} pages)")
    return total_pages


def main():
    parser = argparse.ArgumentParser(description="Split scanned book spreads into individual pages")
    parser.add_argument("pdfs", nargs="*", help="Specific PDFs to process")
    parser.add_argument("--all", action="store_true", help="Process all known spread-scanned books")
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI (default 300)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.all:
        pdfs = [PROCESSED_DIR / name for name in PROBLEM_BOOKS]
    elif args.pdfs:
        pdfs = [Path(p) for p in args.pdfs]
    else:
        parser.print_help()
        sys.exit(1)

    total = 0
    for pdf in pdfs:
        if not pdf.exists():
            log.warning(f"Not found: {pdf}")
            continue

        # Output name: strip .bak, keep original name
        out_name = pdf.name.replace(".bak", "")
        output_path = OUTPUT_DIR / out_name
        pages = process_spread_pdf(pdf, output_path, dpi=args.dpi)
        total += pages

    log.info(f"\nDone: {total} individual pages from {len(pdfs)} books")
    log.info(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
