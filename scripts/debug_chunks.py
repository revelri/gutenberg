#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "services" / "worker"))

from pipeline.chunker import _token_count, _split_by_headers, _recursive_split, CHUNK_SIZE, CHUNK_OVERLAP
import fitz

path = Path.home() / "Downloads" / "Xanathar's Lost Notes to Everything Else v1.1.pdf"
doc = fitz.open(str(path))
text = "\n\n".join(p.get_text("text") for p in doc if p.get_text("text").strip())
doc.close()

print(f"Total: {len(text):,} chars, {_token_count(text):,} tokens")
sections = _split_by_headers(text)
print(f"Sections: {len(sections)}")
for i, s in enumerate(sections[:5]):
    print(f"  [{i}] heading={s['heading']!r} tokens={_token_count(s['text']):,}")

chunks = _recursive_split(sections[0]["text"], CHUNK_SIZE, CHUNK_OVERLAP)
print(f"\nChunks from section[0]: {len(chunks)}")
sizes = [_token_count(c) for c in chunks]
print(f"Token sizes (first 20): {sizes[:20]}")
print(f"Min={min(sizes)}, Max={max(sizes)}, Avg={sum(sizes)//len(sizes)}")
