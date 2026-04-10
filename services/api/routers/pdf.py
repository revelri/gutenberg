"""PDF serving router — file delivery, page image rendering."""

import re
from pathlib import Path

import fitz
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse

router = APIRouter()

DATA_DIR = "/data"


def _sanitize_filename(filename: str) -> str:
    if (
        "/" in filename
        or "\\" in filename
        or ".." in filename
        or "\0" in filename
        or re.search(r"%2[fF]|%5[cC]", filename)
    ):
        raise HTTPException(status_code=400, detail="Invalid filename")
    return filename


def _find_pdf(filename: str) -> Path | None:
    safe = _sanitize_filename(filename)
    base = Path(DATA_DIR)
    for subdir in ("processed", "inbox"):
        p = base / subdir / safe
        if p.is_file():
            return p
    return None


@router.get("/api/pdf/{filename}")
async def serve_pdf(filename: str):
    pdf_path = _find_pdf(filename)
    if not pdf_path:
        raise HTTPException(status_code=404, detail="PDF not found")
    return FileResponse(
        str(pdf_path),
        media_type="application/pdf",
        headers={"Accept-Ranges": "bytes"},
    )


@router.get("/api/pdf/{filename}/page/{page_num}/image")
async def page_image(
    filename: str,
    page_num: int,
    highlight: str = Query(default=""),
    dpi: int = Query(default=200),
):
    pdf_path = _find_pdf(filename)
    if not pdf_path:
        raise HTTPException(status_code=404, detail="PDF not found")

    doc = fitz.open(str(pdf_path))
    if page_num < 1 or page_num > len(doc):
        doc.close()
        raise HTTPException(status_code=404, detail="Page out of range")

    page = doc[page_num - 1]

    if highlight:
        rects = page.search_for(highlight)
        if rects:
            page.add_highlight_annot(rects)

    pix = page.get_pixmap(dpi=dpi)
    doc.close()

    jpeg_bytes = pix.tobytes("jpeg")
    return StreamingResponse(
        iter([jpeg_bytes]),
        media_type="image/jpeg",
    )
