#!/usr/bin/env python3
"""Manual batch ingestion CLI — ingest files without the watcher."""

import argparse
import shutil
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Manually ingest documents into Gutenberg")
    parser.add_argument("files", nargs="+", help="PDF or DOCX files to ingest")
    parser.add_argument("--inbox", default="./data/inbox", help="Inbox directory (default: ./data/inbox)")
    args = parser.parse_args()

    inbox = Path(args.inbox)
    inbox.mkdir(parents=True, exist_ok=True)

    for f in args.files:
        src = Path(f)
        if not src.exists():
            print(f"Skipping {f}: file not found", file=sys.stderr)
            continue
        if src.suffix.lower() not in {".pdf", ".docx"}:
            print(f"Skipping {f}: unsupported file type", file=sys.stderr)
            continue

        dest = inbox / src.name
        shutil.copy2(str(src), str(dest))
        print(f"Copied {src.name} → {dest}")

    print(f"\nFiles copied to {inbox}. The worker will process them automatically.")


if __name__ == "__main__":
    main()
