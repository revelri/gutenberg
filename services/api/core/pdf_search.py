import re
from pathlib import Path

import fitz

DATA_DIR = "/data"


def find_quote_page(
    filename: str,
    quote: str,
    search_dirs: list[str] | None = None,
) -> int | None:
    if search_dirs is None:
        search_dirs = ["processed", "inbox"]

    base = Path(DATA_DIR)
    pdf_path = None
    for subdir in search_dirs:
        p = base / subdir / filename
        if p.is_file():
            pdf_path = p
            break
    if pdf_path is None:
        return None

    doc = fitz.open(str(pdf_path))
    try:
        normalized = re.sub(r"\s+", " ", quote.lower().strip())
        for i in range(len(doc)):
            page = doc[i]
            prefix_len = len(normalized)
            while prefix_len >= 10:
                search_term = normalized[:prefix_len]
                if page.search_for(search_term):
                    return i + 1
                prefix_len = int(prefix_len * 0.8)
                if prefix_len < 10:
                    break
        return None
    finally:
        doc.close()
