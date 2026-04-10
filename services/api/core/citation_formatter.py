"""Citation formatter — MLA, APA, ASA, Chicago, Harvard, Sage.

Pure functions: citation data in, formatted string out. No external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Style(str, Enum):
    MLA = "mla"
    APA = "apa"
    CHICAGO = "chicago"
    HARVARD = "harvard"
    ASA = "asa"
    SAGE = "sage"


@dataclass
class Citation:
    """Raw citation data from the RAG pipeline."""

    quote: str
    author: str  # "Deleuze, Gilles" or "Deleuze, Gilles and Felix Guattari"
    title: str  # "A Thousand Plateaus"
    year: int
    page: int | str  # int for PDF, "ch. 3" for EPUB
    publisher: str = ""
    city: str = ""
    editor: str = ""
    doc_type: str = "book"  # book|chapter|article


def format_inline(citation: Citation, style: Style) -> str:
    """Format an in-text parenthetical citation."""
    if not citation.author:
        last = "Unknown"
    else:
        last = _last_name(citation.author)
    multi = _is_multi_author(citation.author)
    pg = _page_str(citation.page)

    if not citation.year:
        year_str = "n.d."
    else:
        year_str = str(citation.year)

    match style:
        case Style.MLA:
            return (
                f"({last} {pg})"
                if not multi
                else f"({_mla_authors(citation.author)} {pg})"
            )

        case Style.APA:
            authors = _apa_authors(citation.author)
            return f"({authors}, {year_str}, p. {pg})"

        case Style.CHICAGO:
            authors = _chicago_note_authors(citation.author)
            return f"{authors}, *{citation.title}*, {pg}."

        case Style.HARVARD:
            authors = _apa_authors(citation.author)
            return f"({authors} {year_str}, p. {pg})"

        case Style.ASA:
            authors = _mla_authors(citation.author)
            return f"({authors} {year_str}:{pg})"

        case Style.SAGE:
            authors = _mla_authors(citation.author)
            return f"({authors}, {year_str}, p. {pg})"


def format_bibliography(citation: Citation, style: Style) -> str:
    """Format a full bibliography/works cited entry."""
    match style:
        case Style.MLA:
            return _bib_mla(citation)
        case Style.APA:
            return _bib_apa(citation)
        case Style.CHICAGO:
            return _bib_chicago(citation)
        case Style.HARVARD:
            return _bib_harvard(citation)
        case Style.ASA:
            return _bib_asa(citation)
        case Style.SAGE:
            return _bib_sage(citation)


# ── Bibliography formatters ──────────────────────────────────────────


def _bib_mla(c: Citation) -> str:
    """MLA 9th edition."""
    authors = _mla_bib_authors(c.author)
    if c.doc_type == "chapter" and c.editor:
        return (
            f"{authors}. \u201c{c.title}.\u201d *{c.editor}*, {c.publisher}, {c.year}."
        )
    return f"{authors}. *{c.title}*. {c.publisher}, {c.year}."


def _bib_apa(c: Citation) -> str:
    """APA 7th edition."""
    authors = _apa_bib_authors(c.author)
    if c.doc_type == "chapter" and c.editor:
        return (
            f"{authors} ({c.year}). {c.title}. In {c.editor}, "
            f"*{c.title}*. {c.publisher}."
        )
    return f"{authors} ({c.year}). *{c.title}*. {c.publisher}."


def _bib_chicago(c: Citation) -> str:
    """Chicago 17th edition (notes-bibliography)."""
    authors = _chicago_bib_authors(c.author)
    if c.city:
        return f"{authors}. *{c.title}*. {c.city}: {c.publisher}, {c.year}."
    return f"{authors}. *{c.title}*. {c.publisher}, {c.year}."


def _bib_harvard(c: Citation) -> str:
    """Harvard style."""
    authors = _apa_bib_authors(c.author)
    if c.city:
        return f"{authors} ({c.year}) *{c.title}*. {c.city}: {c.publisher}."
    return f"{authors} ({c.year}) *{c.title}*. {c.publisher}."


def _bib_asa(c: Citation) -> str:
    """ASA (American Sociological Association) 6th edition."""
    authors = _apa_bib_authors(c.author)
    # Avoid double period after initials
    sep = " " if authors.endswith(".") else ". "
    if c.city:
        return f"{authors}{sep}{c.year}. *{c.title}*. {c.city}: {c.publisher}."
    return f"{authors}{sep}{c.year}. *{c.title}*. {c.publisher}."


def _bib_sage(c: Citation) -> str:
    """SAGE Harvard variant (follows APA closely)."""
    return _bib_apa(c)


# ── Author name helpers ──────────────────────────────────────────────


def _parse_authors(author: str) -> list[str]:
    """Split 'Deleuze, Gilles and Felix Guattari' into individual names."""
    parts = [
        a.strip() for a in author.replace(" and ", ",").replace(" & ", ",").split(",")
    ]
    # Re-pair: "Deleuze", "Gilles", "Felix Guattari" → ["Deleuze, Gilles", "Felix Guattari"]
    authors = []
    i = 0
    while i < len(parts):
        if i + 1 < len(parts) and not " " in parts[i] and not " " in parts[i + 1]:
            # "Deleuze", "Gilles" → "Deleuze, Gilles"
            authors.append(f"{parts[i]}, {parts[i + 1]}")
            i += 2
        else:
            authors.append(parts[i])
            i += 1
    return authors


def _last_name(author: str) -> str:
    """Extract first author's last name."""
    if "," in author:
        return author.split(",")[0].strip()
    return author.split()[-1] if author else ""


def _is_multi_author(author: str) -> bool:
    return " and " in author.lower() or " & " in author.lower()


def _mla_authors(author: str) -> str:
    """MLA in-text: 'Deleuze and Guattari'."""
    authors = _parse_authors(author)
    if len(authors) == 1:
        return _last_name(authors[0])
    if len(authors) == 2:
        return f"{_last_name(authors[0])} and {_last_name(authors[1])}"
    return f"{_last_name(authors[0])} et al."


def _apa_authors(author: str) -> str:
    """APA in-text: 'Deleuze & Guattari'."""
    if not author:
        return "Unknown"
    authors = _parse_authors(author)
    if len(authors) == 1:
        return _last_name(authors[0])
    if len(authors) == 2:
        return f"{_last_name(authors[0])} & {_last_name(authors[1])}"
    return f"{_last_name(authors[0])} et al."


def _chicago_note_authors(author: str) -> str:
    """Chicago notes: 'Gilles Deleuze and Felix Guattari'."""
    authors = _parse_authors(author)
    formatted = []
    for a in authors:
        if ", " in a:
            last, first = a.split(", ", 1)
            formatted.append(f"{first} {last}")
        else:
            formatted.append(a)
    if len(formatted) == 1:
        return formatted[0]
    if len(formatted) == 2:
        return f"{formatted[0]} and {formatted[1]}"
    return f"{formatted[0]} et al."


def _mla_bib_authors(author: str) -> str:
    """MLA bibliography: 'Deleuze, Gilles, and Felix Guattari'."""
    authors = _parse_authors(author)
    if len(authors) == 1:
        return authors[0]
    if len(authors) == 2:
        second = authors[1]
        if ", " in second:
            last, first = second.split(", ", 1)
            second = f"{first} {last}"
        return f"{authors[0]}, and {second}"
    return f"{authors[0]}, et al."


def _apa_bib_authors(author: str) -> str:
    """APA bibliography: 'Deleuze, G., & Guattari, F.'."""
    authors = _parse_authors(author)
    formatted = []
    for a in authors:
        if ", " in a:
            last, first = a.split(", ", 1)
            initials = ". ".join(w[0] for w in first.split() if w) + "."
            formatted.append(f"{last}, {initials}")
        elif " " in a:
            # "Felix Guattari" → "Guattari, F."
            parts = a.split()
            last = parts[-1]
            initials = ". ".join(p[0] for p in parts[:-1]) + "."
            formatted.append(f"{last}, {initials}")
        else:
            formatted.append(a)
    if len(formatted) == 1:
        return formatted[0]
    if len(formatted) == 2:
        return f"{formatted[0]}, & {formatted[1]}"
    return f"{formatted[0]}, et al."


def _chicago_bib_authors(author: str) -> str:
    """Chicago bibliography: 'Deleuze, Gilles, and Felix Guattari'."""
    return _mla_bib_authors(author)  # Same format


def _page_str(page: int | str) -> str:
    """Convert page to string, handling EPUB chapters."""
    if isinstance(page, str):
        return page
    return str(page)
