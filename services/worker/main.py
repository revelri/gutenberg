"""Gutenberg Worker — watches inbox and ingests documents."""

import logging
import os
import sys
import time

from pipeline.watcher import InboxWatcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("gutenberg.worker")


def wait_for_services():
    """Block until Ollama and ChromaDB are reachable."""
    import httpx

    ollama_url = os.environ.get("OLLAMA_HOST", "http://ollama:11434")
    chroma_url = os.environ.get("CHROMA_HOST", "http://chromadb:8000")

    for name, url in [("Ollama", ollama_url), ("ChromaDB", chroma_url)]:
        for attempt in range(60):
            try:
                r = httpx.get(f"{url}/api/version" if name == "Ollama" else f"{url}/api/v1/heartbeat", timeout=5)
                if r.status_code == 200:
                    log.info(f"{name} is ready")
                    break
            except Exception:
                pass
            if attempt % 10 == 0:
                log.info(f"Waiting for {name} at {url}...")
            time.sleep(2)
        else:
            log.error(f"{name} not reachable after 120s, starting anyway")


def main():
    log.info("Gutenberg Worker starting...")
    wait_for_services()

    inbox = os.environ.get("INBOX_DIR", "/data/inbox")
    os.makedirs(inbox, exist_ok=True)

    watcher = InboxWatcher(inbox)
    log.info(f"Watching {inbox} for new documents")
    watcher.run()


if __name__ == "__main__":
    main()
