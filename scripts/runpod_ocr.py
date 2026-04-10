#!/usr/bin/env python3
"""RunPod OCR batch: spin up A40, run Surya/Marker on scanned PDFs, download results.

Automates the full lifecycle: create pod → upload → OCR → download → terminate.
Total cost: ~$0.15-0.20 for 7 books.

Usage:
    export RUNPOD_API_KEY=your_key_here
    uv run scripts/runpod_ocr.py                           # OCR all image-only PDFs
    uv run scripts/runpod_ocr.py --pdf "data/processed/1966 Bergsonism*.pdf"  # specific file
    uv run scripts/runpod_ocr.py --dry-run                 # show what would run
"""

import argparse
import glob
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import runpod

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("runpod_ocr")

# ── Config ──────────────────────────────────────────────────────────
SSH_KEY = os.path.expanduser("~/.ssh/id_ed25519")
SSH_PUB = Path(f"{SSH_KEY}.pub").read_text().strip()
REMOTE_WORK = "/workspace"

# Image-only PDFs that need OCR (restore from .bak if available)
IMAGE_ONLY_BOOKS = [
    "1966 Bergsonism - Deleuze, Gilles.pdf",
    "1969 The Logic of Sense - Deleuze, Gilles.pdf",
    "1970 Spinoza Practical Philosophy - Deleuze, Gilles.pdf",
    "1977 Dialogues - Deleuze, Gilles.pdf",
    "1986 Cinema 1 The Movement-Image - Deleuze, Gilles.pdf",
    "1988 The Fold Leibniz and the Baroque - Deleuze, Gilles.pdf",
    "1989 Cinema 2 The Time-Image - Deleuze, Gilles.pdf",
]

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OCR_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "ocr_processed"


def get_source_pdfs(specific_pdfs=None):
    """Get list of PDFs to OCR. Prefers .bak (original scans) over OCRmyPDF'd versions."""
    if specific_pdfs:
        return [Path(p) for p in specific_pdfs]

    pdfs = []
    for name in IMAGE_ONLY_BOOKS:
        bak = PROCESSED_DIR / f"{name}.bak"
        orig = PROCESSED_DIR / name
        if bak.exists():
            pdfs.append(bak)
        elif orig.exists():
            pdfs.append(orig)
        else:
            log.warning(f"Not found: {name}")
    return pdfs


def ssh_cmd(host, port):
    return f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p {port} -i {SSH_KEY} root@{host}"


def scp_cmd(host, port):
    return f"scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P {port} -i {SSH_KEY}"


def run_remote(host, port, cmd, timeout=600):
    full = f'{ssh_cmd(host, port)} "{cmd}"'
    log.info(f"  > {cmd[:100]}")
    result = subprocess.run(full, shell=True, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0 and result.stderr:
        log.warning(f"  stderr: {result.stderr[:300]}")
    if result.stdout:
        for line in result.stdout.strip().split("\n")[-5:]:
            log.info(f"  < {line[:120]}")
    return result


def main():
    parser = argparse.ArgumentParser(description="RunPod OCR batch processing")
    parser.add_argument("--pdf", nargs="*", help="Specific PDFs to OCR")
    parser.add_argument("--gpu", default="NVIDIA RTX A6000", help="GPU type (default: NVIDIA RTX A6000)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-pod", action="store_true", help="Don't terminate pod after")
    args = parser.parse_args()

    api_key = os.environ.get("RUNPOD_API_KEY", "")
    if not api_key:
        log.error("RUNPOD_API_KEY not set. Get one at https://www.runpod.io/console/user/settings")
        sys.exit(1)

    runpod.api_key = api_key

    # Check for unified input directory first
    all_input = Path(__file__).resolve().parent.parent / "data" / "surya_all_input"
    if all_input.exists() and any(all_input.glob("*.pdf")) and not args.pdf:
        pdfs = sorted(all_input.glob("*.pdf"))
        log.info(f"Using unified input from data/surya_all_input/ ({len(pdfs)} PDFs)")
    else:
        pdfs = get_source_pdfs(args.pdf)
    if not pdfs:
        log.error("No PDFs to process")
        sys.exit(1)

    total_size_mb = sum(p.stat().st_size for p in pdfs) / 1024 / 1024
    log.info(f"PDFs to OCR: {len(pdfs)} ({total_size_mb:.0f} MB)")
    for p in pdfs:
        log.info(f"  - {p.name}")

    if args.dry_run:
        print(f"\nDRY RUN — would process {len(pdfs)} PDFs on {args.gpu}")
        print(f"Estimated cost: ~${total_size_mb * 0.001 + 0.15:.2f}")
        return

    t0 = time.time()

    # ── 1. Create pod ──────────────────────────────────────────────
    log.info("[1/6] Creating pod...")
    pod = runpod.create_pod(
        name="gutenberg-ocr",
        image_name="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        gpu_type_id=args.gpu,
        gpu_count=1,
        volume_in_gb=50,
        container_disk_in_gb=20,
        ports="22/tcp",
        support_public_ip=True,
        start_ssh=True,
        env={"PUBLIC_KEY": SSH_PUB},
    )
    pod_id = pod["id"]
    log.info(f"  Pod ID: {pod_id}")

    try:
        # ── 2. Wait for SSH ────────────────────────────────────────
        log.info("[2/6] Waiting for pod...")
        ssh_host, ssh_port = None, None
        for attempt in range(90):
            try:
                info = runpod.get_pod(pod_id)
            except Exception as e:
                log.debug(f"  get_pod retry: {e}")
                time.sleep(5)
                continue

            runtime = info.get("runtime")
            if runtime and runtime.get("ports"):
                for p in runtime["ports"]:
                    if p.get("privatePort") == 22:
                        ssh_host = p["ip"]
                        ssh_port = p["publicPort"]
                        break
            if ssh_host:
                break
            time.sleep(5)

        if not ssh_host:
            log.error("Pod did not start in time")
            raise RuntimeError("Pod boot timeout")

        log.info(f"  SSH: root@{ssh_host}:{ssh_port}")
        time.sleep(15)  # let sshd fully initialize

        # Test SSH
        for retry in range(5):
            r = run_remote(ssh_host, ssh_port, "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", timeout=30)
            if r.returncode == 0:
                break
            time.sleep(10)

        # ── 3. Upload PDFs ─────────────────────────────────────────
        log.info(f"[3/6] Uploading {len(pdfs)} PDFs...")
        run_remote(ssh_host, ssh_port, f"mkdir -p {REMOTE_WORK}/pdfs {REMOTE_WORK}/output")

        for pdf in pdfs:
            log.info(f"  Uploading: {pdf.name} ({pdf.stat().st_size / 1024 / 1024:.1f} MB)")
            subprocess.run(
                f'{scp_cmd(ssh_host, ssh_port)} "{pdf}" root@{ssh_host}:{REMOTE_WORK}/pdfs/',
                shell=True, check=True, timeout=120,
            )

        # ── 4. Install and run marker-pdf ──────────────────────────
        log.info("[4/6] Installing marker-pdf 0.2.17 + compatible deps...")
        install_cmd = ' && '.join([
            "apt-get update -qq && apt-get install -y -qq libmagic1",
            "pip install 'marker-pdf==0.2.17' 'transformers>=4.40,<4.48' 'surya-ocr==0.6.12' 2>&1 | tail -5",
            "python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'",
        ])
        run_remote(ssh_host, ssh_port, install_cmd, timeout=600)

        log.info("[4/6] Running OCR on each PDF...")
        run_remote(ssh_host, ssh_port, f"ls {REMOTE_WORK}/pdfs/", timeout=30)

        for pdf in pdfs:
            remote_name = pdf.name
            log.info(f"  OCR: {remote_name}")
            # marker_single 0.2.x: positional args (filename, output_dir)
            r = run_remote(ssh_host, ssh_port,
                       f"marker_single '{REMOTE_WORK}/pdfs/{remote_name}' {REMOTE_WORK}/output/ 2>&1 | tail -10",
                       timeout=1200)
            if r.returncode != 0:
                log.warning(f"  marker_single failed for {remote_name}")

        # Check output
        r = run_remote(ssh_host, ssh_port, f"ls -la {REMOTE_WORK}/output/")

        # ── 5. Download results ────────────────────────────────────
        log.info("[5/6] Downloading OCR results...")
        OCR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Marker outputs markdown files — we need those
        # Also check if it creates PDFs with text layers
        r = run_remote(ssh_host, ssh_port,
                       f"find {REMOTE_WORK}/output/ -name '*.md' -o -name '*.pdf' | head -20")

        subprocess.run(
            f'{scp_cmd(ssh_host, ssh_port)} -r root@{ssh_host}:{REMOTE_WORK}/output/ "{OCR_OUTPUT_DIR}/"',
            shell=True, check=True, timeout=300,
        )

        elapsed = time.time() - t0
        log.info(f"\nOCR complete in {elapsed:.0f}s (~{elapsed/60:.1f} min)")
        log.info(f"Results in: {OCR_OUTPUT_DIR}")
        log.info(f"Estimated cost: ~${elapsed / 3600 * 0.35:.2f}")

    finally:
        # ── 6. Terminate pod ───────────────────────────────────────
        if not args.keep_pod:
            log.info("[6/6] Terminating pod...")
            try:
                runpod.terminate_pod(pod_id)
                log.info("  Pod terminated.")
            except Exception as e:
                log.error(f"  Failed to terminate pod {pod_id}: {e}")
                log.error(f"  MANUALLY TERMINATE at https://www.runpod.io/console/pods")
        else:
            log.info(f"[6/6] Pod kept alive: {pod_id}")
            log.info(f"  SSH: ssh -p {ssh_port} -i {SSH_KEY} root@{ssh_host}")
            log.info(f"  Terminate: runpod.terminate_pod('{pod_id}')")


if __name__ == "__main__":
    main()
