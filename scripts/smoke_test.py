#!/usr/bin/env python3
"""End-to-end smoke test for Gutenberg.

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --no-cleanup --verbose
    python scripts/smoke_test.py --api-url http://localhost:8000
"""

import argparse
import io
import json
import logging
import subprocess
import sys
import time

import httpx

log = logging.getLogger("smoke_test")

API_URL = "http://localhost:8000"
STARTUP_TIMEOUT = 30
INGEST_TIMEOUT = 120
LLM_TIMEOUT = 60
POLL_INTERVAL = 2


def run_docker_compose_up(timeout: int = STARTUP_TIMEOUT) -> bool:
    log.info("Starting docker compose...")
    try:
        subprocess.run(
            ["docker", "compose", "up", "-d", "--build"],
            capture_output=True,
            timeout=180,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        log.error(f"docker compose up failed: {e}")
        return False
    return True


def run_docker_compose_down() -> None:
    log.info("Tearing down docker compose...")
    subprocess.run(
        ["docker", "compose", "down", "-v"],
        capture_output=True,
        timeout=60,
    )


def wait_for_health(api_url: str, timeout: int = STARTUP_TIMEOUT) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{api_url}/api/health", timeout=5)
            if r.status_code == 200:
                data = r.json()
                log.info(
                    f"Health: {data.get('status', 'unknown')} checks={data.get('checks', {})}"
                )
                return True
        except httpx.ConnectError:
            pass
        except Exception as e:
            log.debug(f"Health check error: {e}")
        time.sleep(POLL_INTERVAL)
    log.error("Health check timed out")
    return False


def create_corpus(api_url: str) -> str | None:
    log.info("Creating corpus...")
    r = httpx.post(
        f"{api_url}/api/corpus",
        data={"name": "smoke-test-corpus", "tags": "automated-test"},
        timeout=15,
    )
    if r.status_code != 200:
        log.error(f"Create corpus failed: {r.status_code} {r.text}")
        return None
    corpus_id = r.json()["id"]
    log.info(f"Corpus created: {corpus_id}")
    return corpus_id


def generate_test_pdf() -> bytes:
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)

    text = (
        "Gutenberg Smoke Test Document\n\n"
        "This document was generated programmatically for end-to-end testing.\n\n"
        "Chapter 1: Introduction\n\n"
        "Gutenberg is a RAG (Retrieval-Augmented Generation) system designed for "
        "academic citation verification. It processes PDF documents, extracts text "
        "chunks, stores them in a vector database (ChromaDB), and uses an LLM to "
        "answer queries with proper citations.\n\n"
        "Chapter 2: Architecture\n\n"
        "The system consists of three services: a FastAPI server, a ChromaDB instance "
        "for vector storage, and a background worker for document processing. "
        "The API handles corpus management, ingestion, conversations, and RAG chat.\n\n"
        "Chapter 3: Citation Verification\n\n"
        "After generating an LLM response, Gutenberg extracts quoted passages and "
        "verifies them against source text in the PDF. This grounding check ensures "
        "cited quotes appear in the referenced document, reducing hallucination.\n"
    )
    page.insert_text((72, 100), text, fontsize=11, fontname="helv")

    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def upload_pdf(api_url: str, corpus_id: str) -> str | None:
    log.info("Generating and uploading test PDF...")
    pdf_bytes = generate_test_pdf()
    r = httpx.post(
        f"{api_url}/api/corpus/{corpus_id}/ingest",
        files=[("files", ("smoke_test.pdf", pdf_bytes, "application/pdf"))],
        timeout=30,
    )
    if r.status_code != 200:
        log.error(f"Upload failed: {r.status_code} {r.text}")
        return None
    job_id = r.json()["job_id"]
    log.info(f"Upload accepted, job: {job_id}")
    return job_id


def wait_for_ingestion(
    api_url: str, corpus_id: str, timeout: int = INGEST_TIMEOUT
) -> bool:
    log.info(f"Waiting for ingestion (timeout={timeout}s)...")
    deadline = time.monotonic() + timeout
    last_status = ""

    while time.monotonic() < deadline:
        try:
            with httpx.stream(
                "GET",
                f"{api_url}/api/corpus/{corpus_id}/ingest/status",
                timeout=10,
            ) as r:
                for line in r.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = json.loads(line[6:])
                    status = data.get("status", "")
                    if status != last_status:
                        log.info(
                            f"  Ingest: {status} ({data.get('completed_files', '?')}/{data.get('total_files', '?')})"
                        )
                        last_status = status
                    if status == "done":
                        return True
                    if status == "failed":
                        log.error(f"  Ingest failed: {data.get('error', 'unknown')}")
                        return False
        except (httpx.ReadTimeout, httpx.ConnectError):
            time.sleep(POLL_INTERVAL)
            continue
    log.error("Ingestion timed out")
    return False


def create_conversation(api_url: str, corpus_id: str) -> str | None:
    log.info("Creating conversation...")
    r = httpx.post(
        f"{api_url}/api/corpus/{corpus_id}/conversations",
        json={"mode": "general", "title": "Smoke Test Conversation"},
        timeout=15,
    )
    if r.status_code != 200:
        log.error(f"Create conversation failed: {r.status_code} {r.text}")
        return None
    conv_id = r.json()["id"]
    log.info(f"Conversation created: {conv_id}")
    return conv_id


def send_query(api_url: str, conv_id: str, timeout: int = LLM_TIMEOUT) -> dict:
    log.info("Sending query...")
    result = {"response": "", "citations": [], "warnings": []}
    deadline = time.monotonic() + timeout

    try:
        with httpx.stream(
            "POST",
            f"{api_url}/api/conversations/{conv_id}/messages",
            json={
                "content": "What is Gutenberg and how does citation verification work?"
            },
            timeout=timeout,
        ) as r:
            for line in r.iter_lines():
                if time.monotonic() > deadline:
                    log.warning("Query timed out during streaming")
                    break
                if not line.startswith("data: "):
                    continue
                try:
                    data = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                if "content" in data:
                    result["response"] += data["content"]
                if "citations" in data:
                    result["citations"] = data["citations"]
                if "message" in data:
                    result["warnings"].append(data["message"])
    except Exception as e:
        log.error(f"Query failed: {e}")

    log.info(
        f"Response: {len(result['response'])} chars, "
        f"{len(result['citations'])} citations, {len(result['warnings'])} warnings"
    )
    return result


def verify_citations(result: dict) -> bool:
    # Primary: structured citation data; Fallback: regex for citation-like patterns
    text = result["response"].strip()
    if len(text) < 50:
        log.error("Response too short — LLM may have failed")
        return False
    if result["citations"]:
        log.info(f"Citations verified: {len(result['citations'])} entries")
        return True
    import re

    has_citation = bool(
        re.search(r"p\.?\s*\d+|page\s*\d+", text)
        or re.search(r"\[Source|source|citation", text, re.IGNORECASE)
    )
    if has_citation:
        log.info("Response contains citation references (text-based check)")
    else:
        log.warning("No citations found in response or verification data")
    return has_citation


def main():
    parser = argparse.ArgumentParser(description="Gutenberg end-to-end smoke test")
    parser.add_argument("--api-url", default=API_URL, help="API base URL")
    parser.add_argument(
        "--no-cleanup", action="store_true", help="Keep containers running"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug output")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")

    steps = []
    passed = 0
    failed = 0

    def record(name: str, ok: bool):
        nonlocal passed, failed
        icon = "+" if ok else "-"
        log.info(f"  [{icon}] {name}: {'PASS' if ok else 'FAIL'}")
        steps.append((name, ok))
        if ok:
            passed += 1
        else:
            failed += 1

    def cleanup_and_exit():
        print_summary(steps, passed, failed)
        if not args.no_cleanup:
            run_docker_compose_down()
        sys.exit(1 if failed else 0)

    log.info("=" * 60)
    log.info("GUTENBORG SMOKE TEST")
    log.info("=" * 60)

    record("docker compose up", run_docker_compose_up())
    if not steps[-1][1]:
        cleanup_and_exit()

    try:
        record("health check", wait_for_health(args.api_url))
        if not steps[-1][1]:
            return cleanup_and_exit()

        corpus_id = create_corpus(args.api_url)
        record("create corpus", corpus_id is not None)
        if not corpus_id:
            return cleanup_and_exit()

        job_id = upload_pdf(args.api_url, corpus_id)
        record("upload PDF", job_id is not None)
        if not job_id:
            return cleanup_and_exit()

        record("ingestion complete", wait_for_ingestion(args.api_url, corpus_id))
        if not steps[-1][1]:
            return cleanup_and_exit()

        conv_id = create_conversation(args.api_url, corpus_id)
        record("create conversation", conv_id is not None)
        if not conv_id:
            return cleanup_and_exit()

        result = send_query(args.api_url, conv_id)
        record("query response", len(result["response"]) >= 50)
        record("citations present", verify_citations(result))

    except Exception as e:
        log.exception(f"Unexpected error: {e}")
        failed += 1
        steps.append(("unexpected error", False))

    cleanup_and_exit()


def print_summary(steps, passed, failed):
    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {passed} passed, {failed} failed (of {len(steps)})")
    print(f"{'=' * 60}")
    for name, ok in steps:
        print(f"  [{'+' if ok else 'x'}] {name}")
    print(f"{'=' * 60}")
    print(f"  OVERALL: {'PASS' if failed == 0 else 'FAIL'}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
